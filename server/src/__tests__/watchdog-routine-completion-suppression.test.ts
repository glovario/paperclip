import { randomUUID } from "node:crypto";
import { and, eq, sql } from "drizzle-orm";
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import {
  agents,
  agentWakeupRequests,
  companies,
  createDb,
  heartbeatRuns,
  issueComments,
  issueRelations,
  issues,
} from "@paperclipai/db";
import {
  getEmbeddedPostgresTestSupport,
  startEmbeddedPostgresTestDatabase,
} from "./helpers/embedded-postgres.js";
import { heartbeatService } from "../services/heartbeat.ts";
import { hasStructuredCompletionOutput } from "../services/recovery/service.ts";

vi.mock("../telemetry.ts", () => ({
  getTelemetryClient: () => ({ track: vi.fn() }),
}));

vi.mock("@paperclipai/shared/telemetry", async () => {
  const actual = await vi.importActual<typeof import("@paperclipai/shared/telemetry")>(
    "@paperclipai/shared/telemetry",
  );
  return {
    ...actual,
    trackAgentFirstHeartbeat: vi.fn(),
  };
});

vi.mock("../adapters/index.ts", async () => {
  const actual = await vi.importActual<typeof import("../adapters/index.ts")>("../adapters/index.ts");
  return {
    ...actual,
    getServerAdapter: vi.fn(() => ({
      supportsLocalAgentJwt: false,
      execute: vi.fn(async () => ({
        exitCode: 0,
        signal: null,
        timedOut: false,
        errorMessage: null,
        summary: "Acknowledged.",
        provider: "test",
        model: "test-model",
      })),
    })),
  };
});

describe("hasStructuredCompletionOutput", () => {
  it("returns false for empty or whitespace-only body", () => {
    expect(hasStructuredCompletionOutput("")).toBe(false);
    expect(hasStructuredCompletionOutput("   ")).toBe(false);
    expect(hasStructuredCompletionOutput("\n\n")).toBe(false);
  });

  it("returns false when no completion markers are present", () => {
    expect(hasStructuredCompletionOutput("Working on the task now")).toBe(false);
    expect(hasStructuredCompletionOutput("## Progress\n\n- Started implementation")).toBe(false);
    expect(hasStructuredCompletionOutput("Still investigating the issue")).toBe(false);
  });

  it("detects heading-based completion markers", () => {
    expect(hasStructuredCompletionOutput("## Done\n\n- Implemented feature X")).toBe(true);
    expect(hasStructuredCompletionOutput("## Summary\n\n- Task completed successfully")).toBe(true);
    expect(hasStructuredCompletionOutput("## Update\n\n- All work finished")).toBe(true);
    expect(hasStructuredCompletionOutput("## Completed\n\nEverything is deployed")).toBe(true);
    expect(hasStructuredCompletionOutput("### Result\n\nTests pass")).toBe(true);
    expect(hasStructuredCompletionOutput("# Finished\n\nAll items addressed")).toBe(true);
    expect(hasStructuredCompletionOutput("## Output\n\nBuild succeeded")).toBe(true);
  });

  it("detects status-line completion markers", () => {
    expect(hasStructuredCompletionOutput("Done\n\n- Deployed to staging")).toBe(true);
    expect(hasStructuredCompletionOutput("Completed\n\n- All tests passing")).toBe(true);
    expect(hasStructuredCompletionOutput("Finished\n\nNo issues found")).toBe(true);
    expect(hasStructuredCompletionOutput("Task complete\n\n- PR merged")).toBe(true);
    expect(hasStructuredCompletionOutput("Work complete\n\n- Feature shipped")).toBe(true);
  });

  it("returns false when completion markers are present but blocker language exists", () => {
    expect(hasStructuredCompletionOutput("## Done\n\n- Feature X done\n- Blocked by deployment freeze")).toBe(false);
    expect(hasStructuredCompletionOutput("## Summary\n\n- Waiting on review from team lead")).toBe(false);
    expect(hasStructuredCompletionOutput("Done\n\n- Code written but depends on API changes")).toBe(false);
    expect(hasStructuredCompletionOutput("## Update\n\nUnresolved merge conflict in main")).toBe(false);
    expect(hasStructuredCompletionOutput("Completed\n\nCannot proceed without credentials")).toBe(false);
    expect(hasStructuredCompletionOutput("## Summary\n\nWaiting for CI to finish")).toBe(false);
    expect(hasStructuredCompletionOutput("## Done\n\nNeeds approval before deploy")).toBe(false);
  });

  it("is case-insensitive for both markers and blockers", () => {
    expect(hasStructuredCompletionOutput("## DONE\n\n- All good")).toBe(true);
    expect(hasStructuredCompletionOutput("## done\n\n- All good")).toBe(true);
    expect(hasStructuredCompletionOutput("## Done\n\n- BLOCKED BY upstream")).toBe(false);
  });

  it("handles multiline comments with mixed content", () => {
    const completedComment = [
      "## Summary",
      "",
      "Routine execution completed successfully.",
      "",
      "- Ran daily report generation",
      "- Exported 142 records",
      "- Uploaded to S3 bucket",
    ].join("\n");
    expect(hasStructuredCompletionOutput(completedComment)).toBe(true);
  });

  it("handles comments with completion heading mid-body", () => {
    const comment = [
      "Started the daily sync process.",
      "",
      "## Done",
      "",
      "- Synced 50 records from source A",
      "- Synced 30 records from source B",
    ].join("\n");
    expect(hasStructuredCompletionOutput(comment)).toBe(true);
  });
});

const embeddedPostgresSupport = await getEmbeddedPostgresTestSupport();
const describeEmbeddedPostgres = embeddedPostgresSupport.supported ? describe : describe.skip;

if (!embeddedPostgresSupport.supported) {
  console.warn(
    `Skipping embedded Postgres routine completion suppression tests: ${embeddedPostgresSupport.reason ?? "unsupported environment"}`,
  );
}

describeEmbeddedPostgres("watchdog routine completion suppression (integration)", () => {
  let db: ReturnType<typeof createDb>;
  let tempDb: Awaited<ReturnType<typeof startEmbeddedPostgresTestDatabase>> | null = null;

  beforeAll(async () => {
    tempDb = await startEmbeddedPostgresTestDatabase("paperclip-routine-completion-suppression-");
    db = createDb(tempDb.connectionString);
  }, 30_000);

  afterEach(async () => {
    for (let attempt = 0; attempt < 100; attempt += 1) {
      const activeRuns = await db
        .select({ id: heartbeatRuns.id })
        .from(heartbeatRuns)
        .where(eq(heartbeatRuns.status, "running"));
      if (activeRuns.length === 0) break;
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
    await db.execute(sql.raw(`TRUNCATE TABLE "companies" CASCADE`));
  });

  afterAll(async () => {
    await tempDb?.cleanup();
  });

  async function seedStrandedRoutineExecution(opts: {
    status: "todo" | "in_progress";
    retryReason: "assignment_recovery" | "issue_continuation_needed";
    completionComment?: string | null;
    originKind?: string;
  }) {
    const companyId = randomUUID();
    const agentId = randomUUID();
    const runId = randomUUID();
    const wakeupRequestId = randomUUID();
    const issueId = randomUUID();
    const now = new Date("2026-04-30T00:00:00.000Z");
    const issuePrefix = `R${companyId.replace(/-/g, "").slice(0, 6).toUpperCase()}`;

    await db.insert(companies).values({
      id: companyId,
      name: "Routine Co",
      issuePrefix,
      requireBoardApprovalForNewAgents: false,
    });

    await db.insert(agents).values({
      id: agentId,
      companyId,
      name: "RoutineAgent",
      role: "engineer",
      status: "idle",
      adapterType: "codex_local",
      adapterConfig: {},
      runtimeConfig: {},
      permissions: {},
    });

    await db.insert(agentWakeupRequests).values({
      id: wakeupRequestId,
      companyId,
      agentId,
      source: "assignment",
      triggerDetail: "system",
      reason: opts.retryReason === "assignment_recovery"
        ? "issue_assignment_recovery"
        : "issue_continuation_needed",
      payload: { issueId },
      status: "failed",
      runId,
      claimedAt: now,
      finishedAt: new Date("2026-04-30T00:05:00.000Z"),
      error: "run failed",
    });

    await db.insert(heartbeatRuns).values({
      id: runId,
      companyId,
      agentId,
      invocationSource: "assignment",
      triggerDetail: "system",
      status: "failed",
      wakeupRequestId,
      contextSnapshot: {
        issueId,
        taskId: issueId,
        wakeReason: opts.retryReason === "assignment_recovery"
          ? "issue_assignment_recovery"
          : "issue_continuation_needed",
        retryReason: opts.retryReason,
      },
      startedAt: now,
      finishedAt: new Date("2026-04-30T00:05:00.000Z"),
      updatedAt: new Date("2026-04-30T00:05:00.000Z"),
      errorCode: "process_lost",
      error: "run failed",
    });

    await db.insert(issues).values({
      id: issueId,
      companyId,
      title: "Routine execution task",
      status: opts.status,
      priority: "medium",
      assigneeAgentId: agentId,
      assigneeUserId: null,
      checkoutRunId: opts.status === "in_progress" ? runId : null,
      issueNumber: 1,
      identifier: `${issuePrefix}-1`,
      originKind: opts.originKind ?? "routine_execution",
      originId: randomUUID(),
      startedAt: opts.status === "in_progress" ? now : null,
    });

    if (opts.completionComment) {
      await db.insert(issueComments).values({
        companyId,
        issueId,
        authorAgentId: agentId,
        body: opts.completionComment,
      });
    }

    return { companyId, agentId, runId, issueId };
  }

  it("suppresses auto-block for routine execution issue with structured completion comment", async () => {
    const { companyId, issueId } = await seedStrandedRoutineExecution({
      status: "in_progress",
      retryReason: "issue_continuation_needed",
      completionComment: "## Summary\n\n- Daily sync completed\n- 142 records processed",
    });
    const heartbeat = heartbeatService(db);

    const result = await heartbeat.reconcileStrandedAssignedIssues();
    expect(result.routineCompletionSuppressed).toBe(1);
    expect(result.escalated).toBe(0);

    const issue = await db.select().from(issues).where(eq(issues.id, issueId)).then((rows) => rows[0]);
    expect(issue?.status).toBe("in_progress");

    const recoveryIssues = await db
      .select()
      .from(issues)
      .where(and(eq(issues.companyId, companyId), eq(issues.originKind, "stranded_issue_recovery")));
    expect(recoveryIssues).toHaveLength(0);
  });

  it("suppresses auto-block for routine execution todo issue with completion comment", async () => {
    const { issueId } = await seedStrandedRoutineExecution({
      status: "todo",
      retryReason: "assignment_recovery",
      completionComment: "Done\n\n- All items processed successfully",
    });
    const heartbeat = heartbeatService(db);

    const result = await heartbeat.reconcileStrandedAssignedIssues();
    expect(result.routineCompletionSuppressed).toBe(1);
    expect(result.escalated).toBe(0);

    const issue = await db.select().from(issues).where(eq(issues.id, issueId)).then((rows) => rows[0]);
    expect(issue?.status).toBe("todo");
  });

  it("does NOT suppress for routine issue without a completion comment", async () => {
    const { issueId } = await seedStrandedRoutineExecution({
      status: "in_progress",
      retryReason: "issue_continuation_needed",
      completionComment: null,
    });
    const heartbeat = heartbeatService(db);

    const result = await heartbeat.reconcileStrandedAssignedIssues();
    expect(result.routineCompletionSuppressed).toBe(0);
    expect(result.escalated).toBe(1);

    const issue = await db.select().from(issues).where(eq(issues.id, issueId)).then((rows) => rows[0]);
    expect(issue?.status).toBe("blocked");
  });

  it("does NOT suppress for routine issue with blocker language in comment", async () => {
    const { issueId } = await seedStrandedRoutineExecution({
      status: "in_progress",
      retryReason: "issue_continuation_needed",
      completionComment: "## Summary\n\n- Ran sync but blocked by upstream API outage",
    });
    const heartbeat = heartbeatService(db);

    const result = await heartbeat.reconcileStrandedAssignedIssues();
    expect(result.routineCompletionSuppressed).toBe(0);
    expect(result.escalated).toBe(1);

    const issue = await db.select().from(issues).where(eq(issues.id, issueId)).then((rows) => rows[0]);
    expect(issue?.status).toBe("blocked");
  });

  it("does NOT suppress for non-routine issues even with completion comment", async () => {
    const { issueId } = await seedStrandedRoutineExecution({
      status: "in_progress",
      retryReason: "issue_continuation_needed",
      completionComment: "## Summary\n\n- Feature implemented and tested",
      originKind: "manual",
    });
    const heartbeat = heartbeatService(db);

    const result = await heartbeat.reconcileStrandedAssignedIssues();
    expect(result.routineCompletionSuppressed).toBe(0);
    expect(result.escalated).toBe(1);

    const issue = await db.select().from(issues).where(eq(issues.id, issueId)).then((rows) => rows[0]);
    expect(issue?.status).toBe("blocked");
  });

  it("does NOT suppress for routine issue with non-completion comment", async () => {
    const { issueId } = await seedStrandedRoutineExecution({
      status: "in_progress",
      retryReason: "issue_continuation_needed",
      completionComment: "## Progress\n\n- Working on step 2 of 5",
    });
    const heartbeat = heartbeatService(db);

    const result = await heartbeat.reconcileStrandedAssignedIssues();
    expect(result.routineCompletionSuppressed).toBe(0);
    expect(result.escalated).toBe(1);

    const issue = await db.select().from(issues).where(eq(issues.id, issueId)).then((rows) => rows[0]);
    expect(issue?.status).toBe("blocked");
  });
});

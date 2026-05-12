import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

// ---------------------------------------------------------------------------
// Hoisted mocks — defined before any imports so vi.mock can reference them
// ---------------------------------------------------------------------------

const {
  runChildProcess,
  ensureCommandResolvable,
  resolveCommandForLogs,
  shapePaperclipWorkspaceEnvForExecution,
  rewriteWorkspaceCwdEnvVarsForExecution,
  refreshPaperclipWorkspaceEnvForExecution,
  readPaperclipIssueWorkModeFromContext,
  overrideAdapterExecutionTargetRemoteCwd,
  resolveAdapterExecutionTargetTimeoutSec,
  ensureAdapterExecutionTargetRuntimeCommandInstalled,
} = vi.hoisted(() => ({
  runChildProcess: vi.fn(),
  ensureCommandResolvable: vi.fn(async () => undefined),
  resolveCommandForLogs: vi.fn(async () => "claude"),
  shapePaperclipWorkspaceEnvForExecution: vi.fn(() => ({
    workspaceCwd: null,
    workspaceWorktreePath: null,
    workspaceHints: [] as Array<Record<string, unknown>>,
  })),
  rewriteWorkspaceCwdEnvVarsForExecution: vi.fn(() => ({})),
  refreshPaperclipWorkspaceEnvForExecution: vi.fn(),
  readPaperclipIssueWorkModeFromContext: vi.fn(() => null),
  overrideAdapterExecutionTargetRemoteCwd: vi.fn((target: unknown) => target),
  resolveAdapterExecutionTargetTimeoutSec: vi.fn(() => 300),
  ensureAdapterExecutionTargetRuntimeCommandInstalled: vi.fn(async () => undefined),
}));

vi.mock("@paperclipai/adapter-utils/server-utils", async () => {
  const actual = await vi.importActual<typeof import("@paperclipai/adapter-utils/server-utils")>(
    "@paperclipai/adapter-utils/server-utils",
  );
  return {
    ...actual,
    runChildProcess,
    ensureCommandResolvable,
    resolveCommandForLogs,
    shapePaperclipWorkspaceEnvForExecution,
    rewriteWorkspaceCwdEnvVarsForExecution,
    refreshPaperclipWorkspaceEnvForExecution,
    readPaperclipIssueWorkModeFromContext,
  };
});

vi.mock("@paperclipai/adapter-utils/execution-target", async () => {
  const actual = await vi.importActual<typeof import("@paperclipai/adapter-utils/execution-target")>(
    "@paperclipai/adapter-utils/execution-target",
  );
  return {
    ...actual,
    overrideAdapterExecutionTargetRemoteCwd,
    resolveAdapterExecutionTargetTimeoutSec,
    ensureAdapterExecutionTargetRuntimeCommandInstalled,
  };
});

// Stub out prompt-cache so execute() doesn't reach resolvePaperclipInstanceRootForAdapter,
// which is only partially resolvable via vi.importActual in this test environment.
vi.mock("./prompt-cache.js", () => ({
  prepareClaudePromptBundle: vi.fn(async () => ({
    bundleKey: "test-bundle-key",
    rootDir: "/tmp/paperclip-rl-test-bundle",
    addDir: "/tmp/paperclip-rl-test-bundle/add",
    instructionsFilePath: null,
  })),
}));

import { execute } from "./execute.js";

// ---------------------------------------------------------------------------
// Canned process result helpers
// ---------------------------------------------------------------------------

function makeSuccessResult() {
  return {
    exitCode: 0,
    signal: null,
    timedOut: false,
    stdout: [
      JSON.stringify({ type: "system", subtype: "init", session_id: "sess-1", model: "claude-sonnet" }),
      JSON.stringify({
        type: "result",
        session_id: "sess-1",
        result: "ok",
        usage: { input_tokens: 1, cache_read_input_tokens: 0, output_tokens: 1 },
      }),
    ].join("\n"),
    stderr: "",
    pid: 100,
    startedAt: new Date().toISOString(),
  };
}

function make429Result() {
  return {
    exitCode: 1,
    signal: null,
    timedOut: false,
    stdout: "",
    stderr: "HTTP 429: Too Many Requests",
    pid: 101,
    startedAt: new Date().toISOString(),
  };
}

function makeUsageLimitWithResetResult(resetHint: string) {
  return {
    exitCode: 1,
    signal: null,
    timedOut: false,
    stdout: `${resetHint}\n`,
    stderr: "",
    pid: 102,
    startedAt: new Date().toISOString(),
  };
}

function makeAuthErrorResult() {
  return {
    exitCode: 1,
    signal: null,
    timedOut: false,
    stdout: "",
    stderr: "Please log in. Run claude login first.",
    pid: 103,
    startedAt: new Date().toISOString(),
  };
}

function makeMaxTurnsResult() {
  return {
    exitCode: 1,
    signal: null,
    timedOut: false,
    stdout: JSON.stringify({
      type: "result",
      subtype: "error_max_turns",
      result: "Maximum turns reached.",
      is_error: true,
    }),
    stderr: "",
    pid: 104,
    startedAt: new Date().toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Test suite
// ---------------------------------------------------------------------------

describe("claude_local 429 backoff/retry loop", () => {
  const cleanupDirs: string[] = [];

  afterEach(async () => {
    vi.clearAllMocks();
    vi.useRealTimers();
    while (cleanupDirs.length > 0) {
      const dir = cleanupDirs.pop();
      if (dir) await rm(dir, { recursive: true, force: true }).catch(() => undefined);
    }
  });

  // Helper: calls execute() with a minimal local workspace and collectable logs.
  // rateLimitBackoffBaseMs defaults to 0 so sleep(0) is called — instant with no
  // real delay even without fake timers.
  async function runExecute(config: Record<string, unknown> = {}) {
    const workspaceDir = await mkdtemp(path.join(os.tmpdir(), "paperclip-rl-test-"));
    cleanupDirs.push(workspaceDir);
    const logLines: string[] = [];

    const result = await execute({
      runId: "run-rl-test",
      agent: {
        id: "agent-1",
        companyId: "company-1",
        name: "Test Agent",
        adapterType: "claude_local",
        adapterConfig: {},
      },
      runtime: {
        sessionId: null,
        sessionParams: null,
        sessionDisplayId: null,
        taskKey: null,
      },
      config: {
        command: "claude",
        rateLimitMaxRetries: 3,
        rateLimitBackoffBaseMs: 0,   // sleep(0) — effectively instant
        rateLimitBackoffCeilingMs: 300_000,
        ...config,
      },
      context: {
        paperclipWorkspace: { cwd: workspaceDir, source: "project_primary" },
      },
      onLog: async (_stream: string, text: string) => {
        logLines.push(text);
      },
    });

    return { result, logLines, logText: logLines.join("") };
  }

  // -------------------------------------------------------------------------
  // 1. Success on first retry
  // -------------------------------------------------------------------------
  it("returns success when the first attempt is 429 and the second succeeds", async () => {
    runChildProcess
      .mockResolvedValueOnce(make429Result())
      .mockResolvedValueOnce(makeSuccessResult());

    const { result, logText } = await runExecute();

    expect(result.errorCode).not.toBe("rate_limit_exhausted");
    expect(result.errorFamily).not.toBe("transient_upstream");
    expect(runChildProcess).toHaveBeenCalledTimes(2);
    expect(logText).toContain("[paperclip] Transient API error (rate limit/overload); backing off 0s before retry 1/3");
  });

  // -------------------------------------------------------------------------
  // 2. Success on second retry
  // -------------------------------------------------------------------------
  it("returns success when the first two attempts are 429 and the third succeeds", async () => {
    runChildProcess
      .mockResolvedValueOnce(make429Result())
      .mockResolvedValueOnce(make429Result())
      .mockResolvedValueOnce(makeSuccessResult());

    const { result, logText } = await runExecute();

    expect(result.errorCode).not.toBe("rate_limit_exhausted");
    expect(result.errorFamily).not.toBe("transient_upstream");
    expect(runChildProcess).toHaveBeenCalledTimes(3);
    expect(logText).toContain("before retry 1/3");
    expect(logText).toContain("before retry 2/3");
  });

  // -------------------------------------------------------------------------
  // 3. Exhaustion after max retries
  // -------------------------------------------------------------------------
  it("returns rate_limit_exhausted after all attempts (initial + 3 retries) return 429", async () => {
    runChildProcess.mockResolvedValue(make429Result()); // every call returns 429

    const { result, logText } = await runExecute();

    expect(result.errorCode).toBe("rate_limit_exhausted");
    expect(result.errorFamily).toBe("transient_upstream");
    // errorMessage includes the retry count
    expect(result.errorMessage).toMatch(/\b3\b/);
    // 1 initial + 3 retries = 4 total process invocations
    expect(runChildProcess).toHaveBeenCalledTimes(4);
    expect(logText).toContain("before retry 3/3");
  });

  // -------------------------------------------------------------------------
  // 4. Long retryNotBefore passthrough
  // -------------------------------------------------------------------------
  it("does not retry when extractClaudeRetryNotBefore returns a time beyond the ceiling", async () => {
    // Fake the clock to early morning UTC so the "resets 4pm (America/Chicago)"
    // hint is many hours away (>>rateLimitBackoffCeilingMs of 300 000 ms).
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-04-22T08:00:00.000Z")); // 3 am CDT, 13 h before 4 pm CDT

    runChildProcess.mockResolvedValueOnce(
      makeUsageLimitWithResetResult("You're out of extra usage · resets 4pm (America/Chicago)"),
    );

    const { result } = await runExecute();

    // No retry should have been attempted
    expect(runChildProcess).toHaveBeenCalledTimes(1);
    // The transient error is preserved with retryNotBefore set
    expect(result.errorFamily).toBe("transient_upstream");
    expect(result.retryNotBefore).toBeTruthy();
    // Must NOT be upgraded to rate_limit_exhausted (retryCount === 0)
    expect(result.errorCode).not.toBe("rate_limit_exhausted");
  });

  // -------------------------------------------------------------------------
  // 5. Config override: rateLimitMaxRetries: 1
  // -------------------------------------------------------------------------
  it("exhausts after 1 retry when rateLimitMaxRetries is configured to 1", async () => {
    runChildProcess.mockResolvedValue(make429Result());

    const { result } = await runExecute({ rateLimitMaxRetries: 1 });

    expect(result.errorCode).toBe("rate_limit_exhausted");
    // errorMessage uses singular "retry" for exactly 1 retry
    expect(result.errorMessage).toMatch(/\b1\s+retry\b/);
    // 1 initial + 1 retry = 2 total
    expect(runChildProcess).toHaveBeenCalledTimes(2);
  });

  // -------------------------------------------------------------------------
  // 6a. Non-rate-limit: auth error — no backoff attempted
  // -------------------------------------------------------------------------
  it("does not retry on an authentication error", async () => {
    runChildProcess.mockResolvedValueOnce(makeAuthErrorResult());

    const { result, logText } = await runExecute();

    expect(result.errorCode).not.toBe("rate_limit_exhausted");
    expect(runChildProcess).toHaveBeenCalledTimes(1);
    expect(logText).not.toContain("[paperclip] Transient API error");
  });

  // -------------------------------------------------------------------------
  // 6b. Non-rate-limit: max-turns error — no backoff attempted
  // -------------------------------------------------------------------------
  it("does not retry on a max-turns error", async () => {
    runChildProcess.mockResolvedValueOnce(makeMaxTurnsResult());

    const { result, logText } = await runExecute();

    expect(result.errorCode).not.toBe("rate_limit_exhausted");
    expect(runChildProcess).toHaveBeenCalledTimes(1);
    expect(logText).not.toContain("[paperclip] Transient API error");
  });
});

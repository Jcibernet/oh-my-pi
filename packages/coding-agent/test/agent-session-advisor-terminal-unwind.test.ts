import { afterEach, expect, it } from "bun:test";
import { type } from "@oh-my-pi/omptype";
import { Agent, type AgentMessage, type AgentTool } from "@oh-my-pi/pi-agent-core";
import { createMockModel, type MockResponse } from "@oh-my-pi/pi-ai/providers/mock";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { TempDir } from "@oh-my-pi/pi-utils";

type BoundaryTool = AgentTool<any, any, any>;

function textResponse(text: string): MockResponse {
	return { content: [text], stopReason: "stop" };
}

function toolResponse(id: string, name: string, args: Record<string, unknown> = {}): MockResponse {
	return { content: [{ type: "toolCall", id, name, arguments: args }], stopReason: "toolUse" };
}

function contentText(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.map(part => {
			if (typeof part === "string") return part;
			if (part && typeof part === "object" && "text" in part && typeof part.text === "string") return part.text;
			return "";
		})
		.join("\n");
}

function makeTool(name: string, execute: BoundaryTool["execute"]): BoundaryTool {
	return {
		name,
		label: name,
		description: `${name} test tool`,
		parameters: type({}),
		execute,
	};
}

let active: { session: AgentSession; auth: AuthStorage; temp: TempDir } | undefined;

afterEach(async () => {
	await active?.session.dispose().catch(() => {});
	active?.auth.close();
	await active?.temp.remove().catch(() => {});
	active = undefined;
});

it.each(["concern", "nit", "blocker"] as const)(
	"routes late terminal %s correctly before a real next run",
	async severity => {
		const temp = TempDir.createSync("@pi-advisor-terminal-unwind-");
		const auth = await AuthStorage.create(":memory:");
		auth.setRuntimeApiKey("anthropic", "test-key");
		const model = getBundledModel("anthropic", "claude-sonnet-4-5");
		if (!model) throw new Error("Expected bundled model");
		const nextUserMarker = "NEXT_USER_CONTEXT_MARKER";
		const terminalTurnEnd = Promise.withResolvers<void>();
		const advisorStarted = Promise.withResolvers<void>();
		const releaseAdvisor = Promise.withResolvers<void>();
		const adviceAccepted = Promise.withResolvers<{ feedback: string; streaming: boolean }>();
		const nextProviderStarted = Promise.withResolvers<void>();
		const releaseNextProvider = Promise.withResolvers<void>();
		const nextPrimaryCall = severity === "blocker" ? 4 : 3;
		let primaryCalls = 0;
		const primaryContexts: string[] = [];
		let advisorCalls = 0;
		let terminalReleaseStarted = false;

		const primaryMock = createMockModel({
			id: "terminal-unwind-primary",
			provider: "anthropic",
			handler: async () => {
				if (primaryCalls === 1) return toolResponse("step-1", "step");
				if (primaryCalls === 2) return textResponse("terminal answer");
				if (primaryCalls === nextPrimaryCall) {
					nextProviderStarted.resolve();
					await releaseNextProvider.promise;
					return textResponse("next answer");
				}
				return textResponse("continuation answer");
			},
		});
		const advisorMock = createMockModel({
			id: "terminal-unwind-advisor",
			provider: "anthropic",
			handler: async () => {
				if (++advisorCalls === 1) {
					advisorStarted.resolve();
					await releaseAdvisor.promise;
					return toolResponse("advice-1", "advise", {
						note: "late terminal advice",
						severity,
					});
				}
				return textResponse("advisor quiet");
			},
		});
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model,
				systemPrompt: ["terminal unwind regression"],
				tools: [makeTool("step", async () => ({ content: [{ type: "text", text: "step complete" }] }))],
			},
			streamFn: (messages, context, options) => {
				primaryCalls++;
				primaryContexts.push(JSON.stringify(context.messages));
				return primaryMock.stream(messages, context, options);
			},
		});
		const originalSetOnTurnEnd = agent.setOnTurnEnd.bind(agent);
		agent.setOnTurnEnd = callback => {
			if (!callback) {
				originalSetOnTurnEnd(undefined);
				return;
			}
			originalSetOnTurnEnd(async (messages, signal, context) => {
				await callback(messages, signal, context);
				if (context?.willContinue === false && !terminalReleaseStarted) {
					terminalReleaseStarted = true;
					terminalTurnEnd.resolve();
					releaseAdvisor.resolve();
					await adviceAccepted.promise;
				}
			});
		};

		const settings = Settings.isolated({
			"compaction.enabled": false,
			"retry.enabled": false,
			"advisor.syncBacklog": "off",
		});
		settings.setModelRole("advisor", "anthropic/claude-sonnet-4-5");
		const session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settings,
			modelRegistry: new ModelRegistry(auth, temp.join("models.yml")),
			advisorTools: [],
			advisorStreamFn: advisorMock.stream,
		});
		active = { session, auth, temp };
		if (!session.setAdvisorEnabled(true)) throw new Error("Expected advisor runtime");
		let agentStarts = 0;
		const secondAgentStart = Promise.withResolvers<void>();
		session.subscribe(event => {
			if (event.type !== "agent_start") return;
			agentStarts++;
			if (agentStarts === 2) secondAgentStart.resolve();
		});
		const advisor = session.getAdvisorAgent();
		if (!advisor) throw new Error("Expected advisor agent");
		const advise = advisor.state.tools.find(tool => tool.name === "advise");
		if (!advise) throw new Error("Expected advise tool");
		const originalExecute = advise.execute.bind(advise);
		advise.execute = async (...args) => {
			const result = await originalExecute(...args);
			const feedback = contentText(result.content);
			if (/Delivered|Queued|preserved|urgent/i.test(feedback)) {
				adviceAccepted.resolve({ feedback, streaming: agent.state.isStreaming });
			}
			return result;
		};

		const run = session.prompt("run a step then finish");
		await advisorStarted.promise;
		await terminalTurnEnd.promise;
		const accepted = await adviceAccepted.promise;
		await run;
		await session.waitForIdle();

		expect(accepted.streaming).toBe(true);
		const terminalCalls = severity === "blocker" ? 3 : 2;
		expect(primaryCalls).toBe(terminalCalls);
		const cards = session.agent.state.messages.filter(
			(message: AgentMessage) =>
				message.role === "custom" && "customType" in message && message.customType === "advisor",
		);
		expect(cards).toHaveLength(1);
		if (severity !== "blocker") {
			const card = cards[0];
			expect(card?.role).toBe("custom");
			if (card?.role === "custom") expect(contentText(card.content)).toContain("late terminal advice");
		}

		const nextRun = session.prompt(nextUserMarker);
		await secondAgentStart.promise;
		await nextProviderStarted.promise;
		expect(session.agent.state.isStreaming).toBe(true);
		const liveResult = await advise.execute("live-next", {
			note: "live next-turn concern",
			severity: "concern",
		});
		expect(contentText(liveResult.content)).toMatch(/Delivered|Queued/);
		releaseNextProvider.resolve();
		await nextRun;
		await session.waitForIdle();
		expect(agentStarts).toBe(2);
		// The live concern intentionally steers one continuation after the held
		// next-user provider request; this is separate from the terminal-run guard.
		expect(primaryCalls).toBe(terminalCalls + 2);
		expect(primaryContexts[terminalCalls]).toContain(nextUserMarker);
	},
);

// Regression: the first end-of-turn blocker still steers a triggered turn
// (#5628), and that steer arms the immune-turn window — so a second blocker,
// arriving with no human message in between, is downgraded to a non-interrupting
// aside instead of re-triggering yet another primary turn. Before the downgrade,
// blocker #2 recreated the loop: each triggered turn ended in another terminal
// answer for the next blocker to wake.
it("downgrades the second idle end-of-turn blocker to an aside after the first one steered", async () => {
	const temp = TempDir.createSync("@pi-advisor-blocker-immune-");
	const auth = await AuthStorage.create(":memory:");
	auth.setRuntimeApiKey("anthropic", "test-key");
	const model = getBundledModel("anthropic", "claude-sonnet-4-5");
	if (!model) throw new Error("Expected bundled model");

	let primaryCalls = 0;
	const primaryMock = createMockModel({
		id: "blocker-immune-primary",
		provider: "anthropic",
		handler: async () => {
			primaryCalls++;
			if (primaryCalls === 1) return toolResponse("step-1", "step");
			return textResponse(`terminal answer ${primaryCalls}`);
		},
	});
	const advisorMock = createMockModel({
		id: "blocker-immune-advisor",
		provider: "anthropic",
		handler: async () => textResponse("advisor quiet"),
	});
	const agent = new Agent({
		getApiKey: () => "test-key",
		initialState: {
			model,
			systemPrompt: ["blocker immune window regression"],
			tools: [makeTool("step", async () => ({ content: [{ type: "text", text: "step complete" }] }))],
		},
		streamFn: (messages, context, options) => primaryMock.stream(messages, context, options),
	});
	const settings = Settings.isolated({
		"compaction.enabled": false,
		"retry.enabled": false,
		"advisor.syncBacklog": "off",
	});
	settings.setModelRole("advisor", "anthropic/claude-sonnet-4-5");
	const session = new AgentSession({
		agent,
		sessionManager: SessionManager.inMemory(),
		settings,
		modelRegistry: new ModelRegistry(auth, temp.join("models.yml")),
		advisorTools: [],
		advisorStreamFn: advisorMock.stream,
	});
	active = { session, auth, temp };
	if (!session.setAdvisorEnabled(true)) throw new Error("Expected advisor runtime");

	let agentStarts = 0;
	const secondAgentStart = Promise.withResolvers<void>();
	session.subscribe(event => {
		if (event.type !== "agent_start") return;
		agentStarts++;
		if (agentStarts === 2) secondAgentStart.resolve();
	});
	const advisor = session.getAdvisorAgent();
	if (!advisor) throw new Error("Expected advisor agent");
	const advise = advisor.state.tools.find(tool => tool.name === "advise");
	if (!advise) throw new Error("Expected advise tool");

	// Turn 1 ends in a terminal answer; idle, no queued work.
	await session.prompt("run a step then finish");
	await session.waitForIdle();
	expect(agentStarts).toBe(1);

	// Blocker #1: idle + terminal tail + immune window not yet armed -> steers a
	// triggered turn (#5628) and arms the immune window on delivery.
	const first = await advise.execute("blocker-1", { note: "first end-of-turn blocker", severity: "blocker" });
	expect(contentText(first.content)).toContain("Delivered");
	await secondAgentStart.promise;
	await session.waitForIdle();
	expect(agentStarts).toBe(2);
	const turn2Calls = primaryCalls;

	// Blocker #2: still no human message, immune window active -> aside, no new turn.
	const second = await advise.execute("blocker-2", { note: "second end-of-turn blocker", severity: "blocker" });
	expect(contentText(second.content)).toContain("Delivered");
	await session.waitForIdle();
	expect(agentStarts).toBe(2);
	expect(primaryCalls).toBe(turn2Calls);

	// Exactly one steered advisor message; the aside has not spawned another.
	const cards = session.agent.state.messages.filter(
		(message: AgentMessage) =>
			message.role === "custom" && "customType" in message && message.customType === "advisor",
	);
	expect(cards.length).toBe(1);
});

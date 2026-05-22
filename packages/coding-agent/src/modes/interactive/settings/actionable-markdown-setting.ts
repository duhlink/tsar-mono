import type { Settings, SettingsManager } from "../../../core/settings-manager.js";

type MarkdownSettingsWithActions = NonNullable<Settings["markdown"]> & {
	actionableCodeBlocks?: boolean;
};

type SettingsWithActionableMarkdown = Settings & {
	markdown?: MarkdownSettingsWithActions;
};

type SettingsManagerWithPrivatePersistence = {
	globalSettings: SettingsWithActionableMarkdown;
	markModified: (field: keyof Settings, nestedKey?: string) => void;
	save: () => void;
};

export function getActionableMarkdownEnabled(
	settingsManager: Pick<SettingsManager, "getGlobalSettings" | "getProjectSettings">,
): boolean {
	const globalSettings = settingsManager.getGlobalSettings() as SettingsWithActionableMarkdown;
	const projectSettings = settingsManager.getProjectSettings() as SettingsWithActionableMarkdown;

	return projectSettings.markdown?.actionableCodeBlocks ?? globalSettings.markdown?.actionableCodeBlocks ?? false;
}

export function setActionableMarkdownEnabled(
	settingsManager: Pick<SettingsManager, "applyOverrides" | "getGlobalSettings" | "getProjectSettings">,
	enabled: boolean,
): void {
	const internals = getPrivatePersistence(settingsManager);
	if (internals !== undefined) {
		internals.globalSettings.markdown = {
			...(internals.globalSettings.markdown ?? {}),
			actionableCodeBlocks: enabled,
		};
		internals.markModified("markdown", "actionableCodeBlocks");
		internals.save();
		return;
	}

	const globalSettings = settingsManager.getGlobalSettings() as SettingsWithActionableMarkdown;
	const projectSettings = settingsManager.getProjectSettings() as SettingsWithActionableMarkdown;
	settingsManager.applyOverrides({
		markdown: {
			...(globalSettings.markdown ?? {}),
			...(projectSettings.markdown ?? {}),
			actionableCodeBlocks: enabled,
		},
	} as Partial<Settings>);
}

function getPrivatePersistence(value: unknown): SettingsManagerWithPrivatePersistence | undefined {
	if (typeof value !== "object" || value === null) {
		return undefined;
	}

	const candidate = value as Partial<SettingsManagerWithPrivatePersistence>;
	if (
		candidate.globalSettings === undefined ||
		typeof candidate.markModified !== "function" ||
		typeof candidate.save !== "function"
	) {
		return undefined;
	}

	return candidate as SettingsManagerWithPrivatePersistence;
}

/**
 * Tests for Kodo VS Code Extension.
 */

import * as assert from "assert";
import * as vscode from "vscode";

suite("Kodo Extension Test Suite", () => {
  vscode.window.showInformationMessage("Starting Kodo tests...");

  test("Extension should be present", () => {
    assert.ok(vscode.extensions.getExtension("kodo.kodo"));
  });

  test("Extension should activate", async () => {
    const extension = vscode.extensions.getExtension("kodo.kodo");
    if (extension) {
      await extension.activate();
      assert.ok(extension.isActive);
    }
  });

  test("Commands should be registered", async () => {
    const commands = await vscode.commands.getCommands(true);

    assert.ok(commands.includes("kodo.askQuestion"));
    assert.ok(commands.includes("kodo.explainSelection"));
    assert.ok(commands.includes("kodo.findReferences"));
    assert.ok(commands.includes("kodo.analyzeImpact"));
    assert.ok(commands.includes("kodo.showCallGraph"));
    assert.ok(commands.includes("kodo.indexRepository"));
  });

  test("Configuration should have defaults", () => {
    const config = vscode.workspace.getConfiguration("kodo");

    assert.strictEqual(
      config.get("serverUrl"),
      "http://localhost:8000"
    );
    assert.strictEqual(config.get("enableHover"), true);
    assert.strictEqual(config.get("enableCodeLens"), true);
    assert.strictEqual(config.get("autoIndex"), false);
    assert.strictEqual(config.get("maxContextLines"), 50);
  });
});

suite("API Types Test Suite", () => {
  test("EntityType values are correct", () => {
    const validTypes = [
      "function",
      "method",
      "class",
      "module",
      "variable",
      "import",
    ];

    // Type checking at compile time
    const entityType: import("../../api/types").EntityType = "function";
    assert.ok(validTypes.includes(entityType));
  });

  test("QueryType values are correct", () => {
    const validTypes = [
      "explain",
      "find",
      "trace",
      "generate",
      "analyze",
      "general",
    ];

    const queryType: import("../../api/types").QueryType = "explain";
    assert.ok(validTypes.includes(queryType));
  });

  test("ImpactLevel values are correct", () => {
    const validLevels = ["none", "low", "medium", "high", "critical"];

    const impactLevel: import("../../api/types").ImpactLevel = "medium";
    assert.ok(validLevels.includes(impactLevel));
  });
});

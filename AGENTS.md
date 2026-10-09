
# Codex Engineering Rules

## Core Rules
- Never assume how existing code works.
- Read relevant files before suggesting changes.
- Never invent functions, variables, imports, APIs, or dependencies.
- Verify function signatures and call sites.
- Use existing project architecture.
- Prefer minimal, targeted fixes.
- Do not rewrite working code unnecessarily.
- Do not introduce paid services.

## Debugging
1. Read the complete error traceback.
2. Locate the exact failing function.
3. Inspect related functions and dependencies.
4. Identify the root cause.
5. Apply the smallest possible fix.
6. Run syntax checks.
7. Run relevant tests.
8. Report any remaining issues.

## Verification
- Never claim code works without testing.
- Never invent test results.
- Distinguish verified facts from assumptions.
- If a dependency's behavior is uncertain, inspect its installed API or documentation.
- Report tests that could not run.

## Safety
- Never delete existing files without permission.
- Never modify .env or credentials.
- Never download large datasets without permission.
- Never modify unrelated code.
- Never disable SSL verification to fix connection errors.

## Response Format
Always report:
1. Root cause
2. Files modified
3. Changes made
4. Tests executed and results
5. Remaining uncertainties

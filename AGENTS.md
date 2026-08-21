# Repository Instructions

## Protected `main` workflow

- Never commit or push changes directly to `main`.
- Start each change from an up-to-date `main` branch and create a dedicated branch using the `codex/` prefix.
- Keep the branch history linear. Rebase onto the latest `main` when synchronization is needed; do not merge `main` into the working branch.
- Open a pull request targeting `main` for every change.
- Resolve every pull request review conversation before merging.
- Wait for the CodeQL checks to finish successfully. Do not merge while code scanning is pending or failing, and address any blocking CodeQL error or security alert of `high` severity or above.
- Use squash merge by default. Rebase merge is acceptable when preserving individual linear commits is intentional; do not create merge commits.
- Do not force-push to or delete `main`.
- After merging, update the local `main` with a fast-forward-only pull and remove the completed working branch when it is no longer needed.

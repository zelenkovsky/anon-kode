# ANON KODE


https://github.com/user-attachments/assets/7a9253a7-8bb0-40d5-a3f3-5e6096d7c789


Terminal-based AI coding retard that can use any model that supports the OpenAI-style API.

- Fixes your spaghetti code
- Explains wtf that function does
- Runs tests and other bullshit
- Whatever else claude-code can do, depending on the model you use

## HOW TO USE

```
npm install -g anon-kode
cd your-project
kode
```

First time you gotta set up your shit with /config. After that, just start typing.

## Warning

Use at own risk.


## YOUR DATA

- All telemetry removed

## DEVELOPMENT

### CI/CD Workflow

This project uses GitHub Actions for automated versioning, building and publishing:

1. **Release Workflow** - Single workflow that handles everything
   - Trigger manually from GitHub Actions tab
   - Choose version type: patch, minor, or major
   - The workflow will:
     - Bump the version in package.json
     - Build the project
     - Commit changes and create a tag
     - Publish to npm

To release a new version:
1. Go to Actions → Release → Run workflow
2. Select version type (patch/minor/major)
3. Click "Run workflow"

Note: Requires `NPM_TOKEN` secret to be set in repository settings.

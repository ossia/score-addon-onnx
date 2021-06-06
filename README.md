# Template add-on for score processes

- Read the plug-in docs [here](https://ossia.io/score-docs/development).
- When cloning this template for the first time, run :

```bash
./init.sh YourAddonName
git add .
git commit -am "Initialize addon"
```

in order to update the namespace, filenames, etc to match your name, as well as generate unique identifiers for your classes.

- Add a Github Access Token named `GITHUB_TOKEN` for the Github actions to work correctly (a zip file will be released when tagging a version). See .github/workflows/release.yml for more information.

- When it is ready and you want to make it available to all _ossia score_ users, register your add-on by submitting a pull-request to https://github.com/ossia/score-addons

- Do not forget to regularly update the versions of score you are testing against by updating .github/workflows/jit.yml

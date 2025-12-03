# Typst Util

To run this utility:
+ [The data folder](src/utils/typst/data) is for the json data, it can hold multiple json files, but the json files need to follow the naming convention, `experiment_name.metric.json`. `experiment_name` and `metric` is used to find the correct files.
+ Have your cli path be at [NAS (root)](/). 
+ Then execute:
    ```sh
    uv run -m src.utils.typst.master OUTPUT_FILE_NAME --experiments EXPERIMENT_NAME --metric METRIC
    ```

Use `--help` to all the arguments: 
```sh
uv run -m src.utils.typst.master --help
```

That is all.



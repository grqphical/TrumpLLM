import pandas as pd

posts = pd.read_json("posts.json", lines=True)
posts = posts[["content"]]

pd.set_option('display.max_colwidth', None)
print(posts.iloc[4])

def clean_content(row):
    prefix = "<p>"
    suffix = "</p>"
    content = row["content"]
    if content.startswith(prefix):
        content = content[len(prefix):]
    if content.endswith(suffix):
        content = content[:-len(suffix)]
    return content

posts["content"] = posts.apply(clean_content, axis=1)
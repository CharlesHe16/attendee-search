# attendee-search

`attendee-search` is a command-line like tool to find EAG attendees with specific interests.

It performs a "fuzzy search" over text fields in SwapCard profiles. This functionality is much better than using the
search features on SwapCard. For some people, it's probably better than using `Ctrl-F` in Excel.

This tool is especially useful for finding attendees with niche interests.

This tool has been created quickly (this afternoon), please contact me if you have any questions or issues.

## Installation

1. Have a `Python 3.7+` environment
2. Install the packages in the `requirements.txt`:
    - Packages include `transformers`, `pandas` and `tqdm`
3. Download the Excel (`*.xlsx`) file from SwapCard that contains attendee data (this should have approximately 1,500 rows)
4. Run `main.py`

Note that the tool expects the Excel file that is downloaded from SwapCard. Don't modify this `.xlsx` file, just
download and move it to the directory with `main.py`. The Excel file is a little idiosyncratic, and this tool expects
the column names to be on a certain row (the 4th row).

## How to use the tool

Run `main.py`. You should get a text prompt, with a few simple instructions.

When you run the tool for the first time, it will generate and save an "embedding file" for the attendee data.

I haven't included screenshots in this public repo, because I don't want to show attendee information. If you're an EA
who wants help or more information, please contact me (e.g. using the communication method that led you to this repo).

## Drawbacks

Compared to the alternatives, the tool works well in finding attendees with more niche interests.

However, right now, the tool isn't very useful if you're using a query like `AI safety` or `animal welfare` that produce
a lot of results. When a search involves canonical EA categories and hundreds of results, I think the value of the
current tool over SwapCard or Excel is uncertain.

While the tool reliably returns results closely related to many simple queries, queries that include a number of
unrelated words can produce inconsistent results. Entering multiple minor variations of a query is useful e.g. searching
for `spanish`, `spanish speakers` and `spanish outreach` can help make sure more attendees with these interests are
found.

## Possibilities for development

There’s many improvements that could be made:

1. **Basic preprocessing**. I just passed entire text fields from SwapCard into the embedder, instead of trying to parse
   sentences or paragraphs. Better preprocessing would improve functionality.
2. **Fine tuning or finding better embeddings**. An off the shelf embedding was used. Getting data and fine-tuning a
   model to understand EA semantics and terminology would work better. Even just trying out different "off-the-shelf"
   embeddings would probably be useful.
3. **Basic augmentation**. Simple “augmentation” ideas, like using a set of automatically generated similar search
   terms, would make the tool work more robustly.
4. **GUI and installer**. Putting this into a GUI and having an installer for non-development users would be great.

The "machine learning" in this tool is just a semantic search that uses 2 lines from Hugging Face, loading a BERT
embedding. This is  functional, but I’m guessing it’s only about "30% effective" as it could be, and could be greatly
improved with little effort.

## Comments on configuration

Limited configuration settings has been implemented as global variables. A small number of users might find it useful to
change these:

In `main.py`:

```Python
class AttendeeSearch:
    RESULTS_PER_PAGE = 5
    THRESHOLD_COSINE_SCORE = 0.30  # search results with cosine score below this aren't considered

    USE_ANY_XLSX_FILE = True
    ROW_OF_SPREADSHEET_CONTAINING_COLUMN_NAMES = 3
```

## Contact Info

You can reach out to me on Slack or SwapCard. You also find contact information on SwapCard.

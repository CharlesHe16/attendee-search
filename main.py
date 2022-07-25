import time
import pickle
import textwrap
import os
from glob import glob
import math

from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer, util


class AttendeeSearch:
    """ Class that ingests a spreadsheet and performs search on it"""

    # Config settings (in lieu of a config file, hacky)
    RESULTS_PER_PAGE = 5
    THRESHOLD_COSINE_SCORE = 0.30  # search results with cosine score below this aren't considered

    USE_ANY_XLSX_FILE = True
    ROW_OF_SPREADSHEET_CONTAINING_COLUMN_NAMES = 3
    SPREAD_SHEET_PATH = "attendee_sheet.xlsx" # not used

    COLUMNS_TO_USE = ["How Others Can Help Me", "How I Can Help Others"]

    def __init__(self,
                 spreadsheet_path=None,
                 embedder_model_name='sentence-transformers/paraphrase-MiniLM-L6-v2',
                 processed_spreadsheet_with_embeddings_path='processed_spreadsheet_with_embeddings.pkl'):

        self.embedder_model_name = embedder_model_name
        self.processed_spreadsheet_with_embeddings_path = processed_spreadsheet_with_embeddings_path

        self.results_per_page = self.RESULTS_PER_PAGE
        self.threshold_cosine_score = self.THRESHOLD_COSINE_SCORE

        self.columnns_to_use = self.COLUMNS_TO_USE

        self.row_of_spreadsheet_containing_column_names = self.ROW_OF_SPREADSHEET_CONTAINING_COLUMN_NAMES
        self.use_any_xlsx_file = self.USE_ANY_XLSX_FILE

        self.spreadsheet_path = spreadsheet_path

        self.results = []
        self.current_page = 0

        self.load_spreadsheet()

        self.setup_embedder()

        # check if embeddings created already, or else we need to create it
        if os.path.exists(processed_spreadsheet_with_embeddings_path):

            with open(processed_spreadsheet_with_embeddings_path, 'rb') as handle:
                content_with_embeddings = pickle.load(handle)

        else:
            content_with_embeddings = self.construct_content_to_create_embeddings()

        assert len(content_with_embeddings) == len(self.df), "Something's wrong, length of content_with_embeddings does not match length of dataframe."

        self.content_with_embeddings = content_with_embeddings

    def load_spreadsheet(self):

        #  error checking of file:
        if not self.spreadsheet_path and not self.use_any_xlsx_file:
            raise ValueError("No spreadsheet path provided and option to `use xlsx file` is not set.")

        if self.spreadsheet_path and os.path.exists(self.spreadsheet_path) is False:
            raise ValueError(f"Spreadsheet path {self.spreadsheet_path} does not exist")

        if not self.spreadsheet_path:
            found_xlsx_files = glob('*.xlsx')
            if len(found_xlsx_files) == 0:
                raise ValueError("\n\nThe tool didn't find an Excel *.xlsx file in the current directory. \nThe tool needs to have the attendee data as an .xlsx file in the same directly as main.py.")

            self.spreadsheet_path = found_xlsx_files[0]

        print(f'Using file: `{self.spreadsheet_path}`')

        self.df = pd.read_excel(self.spreadsheet_path)

        # set the header row as column names
        self.df.columns = self.df.iloc[self.row_of_spreadsheet_containing_column_names]

        # drop preceding rows, assuming that they are not part of the data
        self.df = self.df.drop(range(self.row_of_spreadsheet_containing_column_names + 1))

        self.df.reset_index(inplace=True, drop=True)

    def setup_embedder(self, device='cpu'):

        start_time = time.time()
        print('Loading sentence transformer embedder...', end='')

        self.embedder = SentenceTransformer(self.embedder_model_name, device=device)

        print(f'done (time taken: {time.time() - start_time:.1f} seconds.)')

    def construct_content_to_create_embeddings(self):
        """ Assembles content for each attendee into a list of sentences to be encoded """

        columns_to_use = self.columnns_to_use

        content_with_embeddings = []

        print('Creating embeddings to perform search (only done once, should load from saved file in the future)')

        start_time = time.time()
        for row_num, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):

            _temp_dict = {}
            _temp_dict['columns_used'] = columns_to_use

            for i, column in enumerate(columns_to_use):

                _temp_dict[f'raw_text_{i}'] = row[column]
                _temp_dict['name'] = row['Name']
                _temp_dict['swapcard'] = row['Swapcard'][0]
                _temp_dict['linkedin'] = row['LinkedIn'][0]

                if isinstance(row[column], str) and len(row[column]) > 5:
                    _temp_dict[f'embedding_{i}'] = self.embedder.encode(row[column], convert_to_tensor=True)
                else:
                    _temp_dict[f'embedding_{i}'] = None

            content_with_embeddings.append(_temp_dict)

        with open(self.processed_spreadsheet_with_embeddings_path, 'wb') as f:
            pickle.dump(content_with_embeddings, f)

        print(f"Embeddings created and saved. Time taken: {time.time() - start_time:.1f} seconds\n")

        return content_with_embeddings

    def perform_search_and_print_results(self, query_string):

        num_results = self.results_per_page

        # get the embedding for the search string
        query_embedding = self.embedder.encode(query_string, convert_to_tensor=True)

        start_time = time.time()

        for i, person_dict in enumerate(self.content_with_embeddings):

            for embedding_number, _ in enumerate(person_dict['columns_used']):
                if person_dict[f'embedding_{embedding_number}'] is not None:
                    _result = util.cos_sim(query_embedding, person_dict[f'embedding_{embedding_number}'])
                    person_dict[f'cosine_similarity_{embedding_number}'] = _result.item()
                else:
                    person_dict[f'cosine_similarity_{embedding_number}'] = -1

        # we have multiple cosine similarities because we used different fields
        # amalgate as the max of the cosine similarities
        for i, person_dict in enumerate(self.content_with_embeddings):
            cosine_similarities = [person_dict[f'cosine_similarity_{i}'] for i in
                                   range(len(person_dict['columns_used']))]
            _max_cosine_similarity = max(cosine_similarities)
            _id_of_max_cosine_similarity = max(range(len(cosine_similarities)), key=cosine_similarities.__getitem__)

            person_dict['max_cosine_similarity'] = _max_cosine_similarity
            person_dict['id_of_max_cosine_similarity'] = _id_of_max_cosine_similarity
            person_dict['raw_text_that_produced_max_result'] = person_dict[f'raw_text_{_id_of_max_cosine_similarity}']
            person_dict['name_of_column_with_max_result'] = person_dict['columns_used'][_id_of_max_cosine_similarity]

        # sort by cosine similarity
        content_sorted_by_embeddings = sorted(self.content_with_embeddings, key=lambda x: x['max_cosine_similarity'],
                                              reverse=True)

        results = []
        for i, person_dict in enumerate(content_sorted_by_embeddings):

            if person_dict['max_cosine_similarity'] < self.threshold_cosine_score:
                break

            results.append(person_dict)

        self.results = results
        self.current_page = 0
        self.max_page_num = math.ceil(len(results) / self.results_per_page)

        # print results
        print(
            f'\n{len(self.results)} results found for query: `{query_string}` with threshold {self.threshold_cosine_score: .2f}'
            f' (Time taken: {time.time() - start_time:.2f} seconds)\n')

    def print_results(self):

        page = self.current_page

        if page >= self.max_page_num:
            print(f'\nAlready at last page, showing results from page {self.max_page_num}:\n')
            page = self.max_page_num-1
            self.current_page = page

        start_i = page * self.results_per_page

        for i, person_dict in enumerate(self.results[start_i:start_i + self.results_per_page]):
            print(
                f'{i + start_i + 1}. {person_dict["name"]:<30} Content score: {person_dict["max_cosine_similarity"]:.2f} (Column: "{person_dict["name_of_column_with_max_result"]}")')
            print(f'   SwapCard: {person_dict["swapcard"]}')
            print(f'   LinkedIn: {person_dict["linkedin"]}')
            print(
                f'   {textwrap.fill(person_dict["raw_text_that_produced_max_result"], width=100, subsequent_indent="   ")}\n')

        self.current_page += 1


if __name__ == '__main__':

    at = AttendeeSearch()

    print('Ready.')

    while True:

        if len(at.results) == 0:
            display_string = '\nEnter search string: (or `q` to quit): '
        else:
            display_string = f'Currently on {at.current_page} of {at.max_page_num} pages ({len(at.results)} results in total).\n'
            display_string += f'Enter `c` to continue to next page, `q` to quit, or any other input to make a new search: '

        user_input_text_from_prompt = input(display_string)

        if user_input_text_from_prompt in ['q', 'Q', 'QUIT']:
            break
        elif user_input_text_from_prompt == 'c':
            at.print_results()
        else:
            at.perform_search_and_print_results(user_input_text_from_prompt)
            at.print_results()


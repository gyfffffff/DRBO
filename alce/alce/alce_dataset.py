from torch.utils.data import Dataset
from loguru import logger
from tqdm import tqdm
from alce.utils import make_demo, get_shorter_text
import numpy as np
from argparse import Namespace
from typing import List, Dict, Any
from transformers.tokenization_utils import PreTrainedTokenizer
import torch


class ALCEDataset(Dataset):
    def __init__(
            self,
            args: Namespace,
            eval_data: List[Dict[str, Any]],
            prompt_data: Dict[str, Any],
            tokenizer: PreTrainedTokenizer,
            mode: str = "train"
    ):
        self.args = args
        self.prompt_data = prompt_data
        self.eval_data = eval_data
        self.head_prompt = self.get_head_prompt()
        self.prepare_eval_data()
        self.length = len(eval_data)
        self.tokenizer = tokenizer
        self.mode = mode
        if mode == "train":
            self.length = int(self.length * 0.8)
            self.eval_data = self.eval_data[:self.length]
        elif mode == "test":
            self.length = int(self.length * 0.2)
            self.eval_data = self.eval_data[-self.length:]

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        messages = [
            {'role': 'system',
             'content': 'You are an intelligent assistant to help me answer questions based on context.'},
            {'role': 'user', 'content': self.eval_data[item]['prompt']},
        ]
        prompt_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        self.eval_data[item]['prompt_ids'] = prompt_ids
        self.eval_data[item]['attention_mask'] = [1] * len(prompt_ids)
        return self.eval_data[item]

    def collate_fn(self, batch):
        if "asqa" in self.args.eval_path:
            qa_pairs = [b_['qa_pairs'] for b_ in batch]
            wikipages = [b_['wikipages'] for b_ in batch]
            annotations = [b_['annotations'] for b_ in batch]
            sample_id = [b_['sample_id'] for b_ in batch]
            question = [b_['question'] for b_ in batch]
            docs = [b_['docs'] for b_ in batch]
            answer = [b_['answer'] for b_ in batch]
            prompt = [b_['prompt'] for b_ in batch]
            attention_mask = [b_['attention_mask'] for b_ in batch]
            prompt_ids = [b_['prompt_ids'] for b_ in batch]

            max_length = max([len(ii) for ii in prompt_ids])
            prompt_ids = [[self.tokenizer.eos_token_id] * max(0, max_length-len(ii)) + ii for ii in prompt_ids]
            attention_mask = [[0] * max(0, max_length-len(ii)) + ii for ii in attention_mask]
            return {
                'qa_pairs': qa_pairs,
                'wikipages': wikipages,
                'annotations': annotations,
                'sample_id': sample_id,
                'question': question,
                'docs': docs,
                'answer': answer,
                'prompt': prompt,
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'prompt_ids': torch.tensor(prompt_ids, dtype=torch.long),
            }
        if "eli5" in self.args.eval_path:
            question = [b_['question'] for b_ in batch]
            question_ctx = [b_['question_ctx'] for b_ in batch]
            answer = [b_['answer'] for b_ in batch]
            claims = [b_['claims'] for b_ in batch]
            docs = [b_['docs'] for b_ in batch]
            prompt = [b_['prompt'] for b_ in batch]
            prompt_ids = [b_['prompt_ids'] for b_ in batch]
            attention_mask = [b_['attention_mask'] for b_ in batch]

            max_length = max([len(ii) for ii in prompt_ids])
            prompt_ids = [[self.tokenizer.eos_token_id] * max(0, max_length-len(ii)) + ii for ii in prompt_ids]
            attention_mask = [[0] * max(0, max_length-len(ii)) + ii for ii in attention_mask]
            return {
                'question': question,
                'question_ctx': question_ctx,
                'docs': docs,
                'answer': answer,
                'claims': claims,
                'prompt': prompt,
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'prompt_ids': torch.tensor(prompt_ids, dtype=torch.long),
            }

    def get_head_prompt(self):
        # Generate the demonstration part
        head_prompt = ""
        train_ids = np.random.choice(len(self.prompt_data["demos"]), self.args.shot, replace=False)
        for i, train_id in enumerate(train_ids):
            train_item = self.prompt_data["demos"][train_id]
            ndoc = self.args.ndoc
            if self.args.no_doc_in_demo:
                ndoc = 0
            elif self.args.fewer_doc_in_demo:
                assert self.args.ndoc_in_demo is not None
                ndoc = self.args.ndoc_in_demo
            if i==0:
                head_prompt += make_demo(
                    train_item, prompt=self.prompt_data["demo_prompt"], ndoc=ndoc,
                    doc_prompt=self.prompt_data["doc_prompt"],
                    instruction=self.prompt_data["instruction"], use_shorter=self.args.use_shorter
                )
            else:
                head_prompt += make_demo(
                    train_item, prompt=self.prompt_data["demo_prompt"], ndoc=ndoc,
                    doc_prompt=self.prompt_data["doc_prompt"],
                    instruction="", use_shorter=self.args.use_shorter
                )
            head_prompt += self.prompt_data["demo_sep"]
        return head_prompt

    def prepare_eval_data(self):
        logger.info("Generating prompts...")
        incomplete_doc_list = 0  # For some questions there might be fewer than ndoc documents
        for idx, eval_item in enumerate(tqdm(self.eval_data)):
            
            self.eval_data[idx]['prompt'] = self.head_prompt + make_demo(
                eval_item, prompt=self.prompt_data["demo_prompt"], ndoc=self.args.ndoc,
                doc_prompt=self.prompt_data["doc_prompt"],
                instruction="", use_shorter=self.args.use_shorter,
                test=True
            )
            doc_list = get_shorter_text(
                eval_item, eval_item["docs"],
                self.args.ndoc,
                self.args.use_shorter
            ) if self.args.use_shorter is not None else eval_item["docs"][:self.args.ndoc]
            if not self.args.retrieve_in_all_docs:
                # If --retrieve_in_all_docs, we keep the original docs and do not trim them by ndoc
                # Otherwise, take the new docs (truncated by ndoc and filtered if using summary/extraction)
                self.eval_data[idx]['docs'] = doc_list
            if len(doc_list) < self.args.ndoc:
                incomplete_doc_list += 1

        if incomplete_doc_list > 0:
            logger.warning(
                f"There are {incomplete_doc_list} questions that have incomplete document list (may due to a lot of "
                f"them are filtered out by summary/extraction).")

        logger.info("Done.")


def main():
    pass


if __name__ == '__main__':
    main()


    """
Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (s
ome of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual
claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sent
ence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.

Question: What causes Bi-polar disorder?

Document [1](Title: Bi-polar disorder | definition of Bi-polar disorder by Medical dictionary): bi-polar disorder | definitio
n of bi-polar disorder by medical dictionary https://medical-dictionary.thefreedictionary.com/bi-polar+disorder (redirected f
rom bi-polar disorder) related to bi-polar disorder: depression bipolar disorder, formerly known as manic depression, is a mo
od disorder that causes radical emotional changes and mood swings, from manic, restless highs to depressive, listless lows. m
ost bipolar individuals experience alternating episodes of mania and depression. bipolar disorder is characterized by alterna
ting manic episodes in which the individual feels abnormally euphoric, optimistic, and energetic and depressive periods in wh
ich the individual feels sad, hopeless, guilty, and sometimes suicidal. manic or depressive periods may last for days, weeks,
 or months
Document [2](Title: Mania and Bi-Polar): can go from depressed to “super happy” all in one day, or even in a few days, does n
ot have a bi-polar disorder Bi-polar looks different depending on the severity of the symptoms. Most bi-polar diagnoses that
are made are for bi-polar 2, with bi-polar 1 being much more rare. Bi-polar 1 is so severe that the individual will have peri
ods of such agitation, or such reckless and seemingly foolish behavior that they put themselves or those around them in dange
r. It is not completely clear what causes bi-polar, but genetics seem to have a large role. The biggest factor
Document [3](Title: Bi-Polar disorder): Bi-Polar disorder Bi-polar is generally a cyclic disease where individuals display de
pressive and elevated episodes at regular intervals. It is a disorder resulting from the imbalance of the chemicals in the br
ain that causes a lot of fluctuations of mood. It is a fact that we all experience happy and sad moods, but people with bi-po
lar disorder experience the changes in mood at an increased level. The cause of this disorder is not known completely. Howeve
r, it is estimated that there are different factors responsible for it. It is often connected to a genetic component. People
suffering from the Bi-polar disorder are

Answer:Bipolar disorder is an emotional disorder that causes extreme mood swings between excitement and depression [1][3]. Th
e spectrum of mood swing may span from days to months [1][2]. We are still not certain of the exact factors that cause such d
isorder, but genetics is considered a major factor [2][3].
   
Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (s
ome of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual
claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sent
ence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.

Question: What, in particular, happened today, that caused the Reddit Gold bar to get to 216%?

Document [1](Title: A Conversation With Reddit Co-Founder Alexis Ohanian): has been very careful about what type of advertisi
ng it allows. I’ve always felt that advertising can be reasonable; it’s just that the bar has been set, and continues to be s
et, so low. What do you make of what happened to Sunil Tripathi, who was wrongly accused by redditors, among others, of being
 the Boston bomber? Whether it’s reddit or Twitter, we as individuals using these platforms have to be responsible for how we
 use them. It’s an awful situation, but it’s not indicative of one platform in particular, it’s indicative of the changing wo
rld we live in and
Document [2](Title: What kind of website should i make to make money online How much money can you make teaching english onli
ne | How to make money online now How to make money online reddit): Manipulate Reddit in order to get views for a particular
website. The guy worked for a site that specialised in a certain technology that happens to have alot of investment and atten
tion going into it right now, and his job was simply to try to get the articles to the top of the relevant sub Reddits and ot
her forums. He didn't go into the specifics of how he tried to do that, but from what I gathered it was mainly using a dozen
or so accounts to get attention/discussions on the thread in order to push the post up Reddit.
Document [3](Title: Transcripts – Page 2 – Physical Gold Fund): is the Chinese requirement), and those bars get shipped over
to China. There has been a lot of speculation that at some point those bars would come back on the market. We’d see big 400 o
unce bars reintroduced into GLD inventory, things like that. It hasn’t happened, so it’s good proof that what’s really happen
ing when the West sells off its gold is that it’s moving through Switzerland into China. An interesting dynamic has been occu
rring since the end of Q4 2016 to today. Yesterday, GLD had its first gold inflows and inventory since November 2nd For those
 not familiar

Answer:
    """

import os
import time

import openai
import torch
import torch.multiprocessing as mp

from prover.utils import AttrDict, MODEL_FORMAT


class GeneratorProcess(mp.Process):
    def __init__(self, local_rank, node_rank, model_path, task_queue, request_statuses, lock, args):
        super().__init__()
        self.local_rank = local_rank
        self.node_rank = node_rank
        self.model_path = model_path
        self.task_queue = task_queue
        self.request_statuses = request_statuses
        self.lock = lock
        self.prompt_func = MODEL_FORMAT[args.mode]['prompt']
        self.output_func = MODEL_FORMAT[args.mode]['output']

    def run(self):
        seed = int(time.time()) % 1000 + (self.node_rank * 8 + self.local_rank) * 1000
        os.environ['LOCAL_RANK'] = str(self.local_rank)
        # llm = LLM(model=self.model_path, max_num_batched_tokens=8192, seed=seed, trust_remote_code=True)

        while True:
            inputs = self.task_queue.get()
            if inputs is None: # Terminate when receiving None
                break
            model_inputs = [
                ''.join([
                    item.get('_extra_header', str()),
                    self.prompt_func(item),
                    item.get('_extra_prompt', str()),
                ]) for _, _, item in inputs
            ]
            # model_outputs = llm.generate(
            #     model_inputs,
            #     self.sampling_params,
            #     use_tqdm=False,
            # )
            MAX_REQUESTS_PER_MINUTE = 10000  # OpenAI API limit
            REQUEST_INTERVAL = 60 / MAX_REQUESTS_PER_MINUTE
            last_request_time = 0
            model_outputs = []
            for input in model_inputs:
                print(f"Input: {input}")
                time_since_last_request = time.time() - last_request_time
                if time_since_last_request < REQUEST_INTERVAL:
                    time.sleep(REQUEST_INTERVAL - time_since_last_request)
                retry_count = 0
                max_retries = 5
                backoff_factor = 2
                while retry_count < max_retries:
                    try:
                        response = openai.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "user", "content": input}]
                        )
                        output = response.choices[0].message.content
                        print(output)
                        model_outputs.append(self.output_func(output))
                        last_request_time = time.time()
                        print(f"Last request time: {last_request_time}")
                        print(f"Output: {model_outputs}")
                        break  # Break out of retry loop on success
                    except openai.RateLimitError as e:
                        retry_count += 1
                        sleep_time = backoff_factor ** retry_count
                        print(f"Rate limit exceeded. Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                    except Exception as e:
                        print(f"OpenAI API error: {e}")
                        model_outputs.append(None)
                        break

            with self.lock:
                for (_, request_id, _), output in zip(inputs, model_outputs):
                    self.request_statuses[request_id] = output

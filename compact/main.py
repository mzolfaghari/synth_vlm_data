import os, argparse, random, time, json
from tqdm import tqdm
from multiprocessing import Manager, Pool
from .processor import process_single_image
from .backends import make_client

def main():
    parser = argparse.ArgumentParser(description='Generate COMPACT compositional questions from images')
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--image_dir', type=str, default='images')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--processes', type=int, default=4)
    parser.add_argument('--print_intermediate', action='store_true')
    # Backend: default to our self-hosted Qwen vLLM (OpenAI-compatible); 'gemini' = upstream path.
    parser.add_argument('--backend', choices=['openai', 'gemini'], default='openai')
    parser.add_argument('--api_key', type=str, default=None, help='required only for --backend gemini')
    parser.add_argument('--base_url', type=str, default=os.environ.get('QA_LLM_BASE_URL'),
                        help='OpenAI-compatible endpoint (default env QA_LLM_BASE_URL)')
    parser.add_argument('--model', type=str, default=os.environ.get('QA_LLM_MODEL'),
                        help='model name for --backend openai (default env QA_LLM_MODEL)')
    args = parser.parse_args()

    if args.backend == 'gemini' and not args.api_key:
        parser.error('--api_key is required when --backend gemini')

    # Picklable client config built inside each worker (multiprocessing) via make_client.
    client_cfg = {'backend': args.backend, 'api_key': args.api_key,
                  'base_url': args.base_url, 'model': args.model}

    # Connectivity check (text-only) against whichever backend was selected.
    client = make_client(**client_cfg)
    print("API check:", client.GenerativeModel('models/gemini-2.0-flash')
          .generate_content(["API connection test"]).text[:200])

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"output_k{args.k}.json")
    with open(output_file, 'w') as f:
        f.write('')

    manager = Manager()
    file_lock = manager.Lock()
    counter = manager.Value('i', 0)

    image_files = [f for f in os.listdir(args.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(image_files)
    image_files = image_files[:args.num_samples]

    process_args = [(f, args.image_dir, client_cfg, output_file, file_lock, counter, args.k, args.print_intermediate) for f in image_files]

    with Pool(processes=args.processes) as pool:
        for _ in tqdm(pool.imap_unordered(process_single_image, process_args), total=len(image_files)):
            pass

if __name__ == "__main__":
    main()

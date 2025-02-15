from shop import remax_main
import os

if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    remax_main.main()

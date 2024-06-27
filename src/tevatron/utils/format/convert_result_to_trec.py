import time
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', type=str, default="dense")
    args = parser.parse_args()


    with open(args.input) as f_in, open(args.output, 'w') as f_out:
        start_time = time.time()
        cur_qid = None
        docid_cache = set()
        rank = 0
        for line in f_in:
            qid, docid, score = line.split()
            if cur_qid != qid:
                cur_qid = qid
                rank = 0
                docid_cache.clear()
            if docid not in docid_cache and rank < 1000:
                docid_cache.add(docid)
                rank += 1
                f_out.write(f'{qid} Q0 {docid} {rank} {score} {args.model}\n')

    end_time = time.time()
    print(f"formating cost: {end_time - start_time} s")
    print("[+]convert finished!!!")


if __name__ == '__main__':
    main()
    # print(1)
    # print(2)
    # print(3)
    # print(4)

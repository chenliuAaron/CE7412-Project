def k_mer_enc(filename: "filename without .fasta suffix", k: "k for k-mer encoding"):
    with open(filename + '.fasta', 'r') as file:
        # read DNA sequences into seqs
        seqs = []
        seq = ""
        for line in file.readlines():
            if line.startswith('>'):
                if len(seq) > 0:
                    seqs.append(seq)
                    seq = ""
            else:
                seq += line.strip('\n')
        if len(seq) > 0:
            seqs.append(seq)

        print('The file %s has %d sequences.\n' % (filename, len(seqs)))

        # encode DNA sequences with k-mer
        encodings = []
        for seq in seqs:
            encoding = []
            code = 0
            for c in seq:
                code *= 4
                if c == 'A':
                    code += 0
                if c == 'C':
                    code += 1
                if c == 'G':
                    code += 2
                if c == 'T':
                    code += 3
                code %= 4 ** k
                encoding.append(code)
            assert len(seq) == len(encoding), 'Error: Unmatched number of characters!'
            encodings.append(encoding)
        assert len(seqs) == len(encodings), 'Error: Unmatched number of sequences!'

        import pickle
        # dump encodings into .pkl files
        with open(filename + '.pkl', 'wb') as pkfile:
            pickle.dump(encodings, pkfile)


def create_datasets(filename: "filename without .pkl suffix", numsamples: "number of samples wanted",
                    training_split=0.8, validation_split=0.1, test_split=0.1):
    import pickle
    # load encodings from .pkl files
    with open(filename + '.pkl', 'rb') as pkfile:
        encodings = pickle.load(pkfile)

    import numpy as np
    if len(encodings) < numsamples:
        encodings = np.random.choice(encodings, numsamples, replace=True)
    else:
        encodings = np.random.choice(encodings, numsamples, replace=False)

    # dump datasets into .pkl files
    with open(filename + '_training.pkl', 'wb') as pkfile:
        pickle.dump(encodings[0: int(len(encodings) * training_split)], pkfile)
    with open(filename + '_validation.pkl', 'wb') as pkfile:
        pickle.dump(encodings[int(len(encodings) * training_split):
                              int(len(encodings) * (training_split + validation_split))], pkfile)
    with open(filename + '_test.pkl', 'wb') as pkfile:
        pickle.dump(encodings[int(len(encodings) * (training_split + validation_split)):], pkfile)


if __name__ == '__main__':
    filenames = ['Dengue-4-sequences', 'Ebola-sequences', 'hepatitis-C-3a-sequences',
                 'influenza-A-sequences', 'mers-sequences', 'SARS-CoV-2-sequences']
    for filename in filenames:
        k_mer_enc(filename, 4)
        create_datasets(filename, numsamples=3000)

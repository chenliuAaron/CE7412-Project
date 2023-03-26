def k_mer_enc(filename: "filename without the .fasta suffix", k: "predefined k for k-mer encoding"):
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
        # load encodings from .pkl files
        with open(filename + '.pkl', 'rb') as pkfile:
            encodings = pickle.load(pkfile)


if __name__ == '__main__':
    # filenames = ['Dengue-4-sequences', 'Ebola-sequences', 'hepatitis-C-3a-sequences',
    #              'influenza-A-sequences', 'mers-sequences', 'SARS-CoV-2-sequences']
    k_mer_enc('Dengue-4-sequences', 4)

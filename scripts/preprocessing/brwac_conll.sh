corpus_file="./data/large_corpora/brwac.conll"
out_file="./data/large_corpora/brwac.txt"

# Set the maximum number of sentences to extract
MAX_SENTS=502800000

awk -v max_sents="$MAX_SENTS" '
BEGIN { sentence=""; num_sentences=0; }

# Skip comments
/^#/ { next; }

# A new sentence starts when the token ID is 1
$1 == "1" {
    if (sentence != "") {
        print sentence;
        sentence = "";
        num_sentences++;
        if (num_sentences >= max_sents) {
            exit;
        }
    }
}

# Extract lemma for valid lines
NF > 1 {
    lemma=$3;
    if (lemma == "_") {
        lemma=$2;
    }
    lemma=tolower(lemma);

    sentence=(sentence == "" ? lemma : sentence " " lemma);
}

END {
    if (sentence != "" && num_sentences < max_sents) {
        print sentence;
    }
}
' "${corpus_file}" > "${out_file}"

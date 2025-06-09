corpus_file="./data/large_corpora/brwac.conll"
out_file="./data/large_corpora/brwac.txt"

perl -i -ne 'print unless /^# /' ${corpus_file}
perl -i -ple 'print "" if $. > 1 && /^1\t/' ${corpus_file}

awk '
BEGIN { sentence=""; }
/^$/ {
    if (sentence != "") {
        print sentence;
        sentence = "";
    }
}
!/^#/ && NF > 0 {
    token=$2;

    sentence=(sentence == "" ? token : sentence " " token);
}
END {
    if (sentence != "") {
        print sentence;
    }
}
' ${corpus_file} > ${out_file}

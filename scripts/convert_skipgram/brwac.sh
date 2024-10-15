perl -i -ne 'print unless /^# /' ./data/brwac.conll
perl -i -ple 'print "" if $. > 1 && /^1\t/' ./data/brwac.conll

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
    lemma=($3 == "_" ? token : $3);
    tag=(token ~ /^[[:punct:]]+$/ ? "PUNCT" : $4);

    text=token "|" lemma "|" tag;
    sentence=(sentence == "" ? text : sentence " " text);
}
END {
    if (sentence != "") {
        print sentence;
    }
}
' "data/brwac.txt" > "data/brwac_awk.txt"

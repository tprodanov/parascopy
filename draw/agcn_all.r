#!/bin/bash

dir="$(dirname "${BASH_SOURCE[0]}")"

usage=$(cat <<-END
Draw agCN for all genes in the directory.
Usage:   $(basename "${BASH_SOURCE[0]}") -i <directory> [-o <directory>] [-@ <threads>] -- <agcn.r arguments>

    -i <dir>, --input <dir>
            Input directory.
    -o <dir>, --output <dir>
            Output directory. Use input/plots/agcn by default.
    -@ <int>, --threads <int>
            Use <int> threads [default: 4].
    -r <regex>, --regex <regex>
            Only draw subdirectories that match the regular expression.

    All arguments after -- are supplied to agcn.r
END
)

input=""
output=""
threads=4
regex=""

while (( "$#" )); do
case "$1" in
    -h|--help|help|"")
        # Print help if there is no arguments, or the first argument is "-h", "--help", "help".
        echo "$usage"
        exit 0
        ;;

    -i|--input)
        input="$2"
        shift 2
        ;;

    -o|--output)
        output="$2"
        shift 2
        ;;

    -@|--threads)
        threads="$2"
        shift 2
        ;;

    -r|--regex)
        regex="$2"
        shift 2
        ;;

    --)
        shift 1
        break
        ;;

    *)
        echo -e "Error: unknown argument $1" >&2
        exit 1
        ;;
esac
done

if [[ ${input} = "" ]]; then
    >&2 echo "Missing -i/--input argument."
fi

if [[ ${output} = "" ]]; then
    output=${input}"/plots/agcn"
fi
mkdir -p "$output"

# ==== Draw plots ====

echo "Drawing agCN:"
echo "    Input:   $input"
echo "    Output:  $output"
echo "    Threads: $threads"
if [[ ${regex} != "" ]]; then
    echo "    Regex:    $regex"
fi

export input="$input"
export output="$output"
export dir="$dir"

gene_name() {
    realpath --relative-to="$input" "$1" | cut -d"/" -f1
}
export -f gene_name

find "$input" -type d -name extra -regex ".*$regex.*" | \
    xargs -i -P "$threads" sh -c 'gene=$(gene_name {}); ${dir}/agcn.r -i {} -o "$output" -g "$gene" '$*

#!/bin/bash

wdir="$(dirname "${BASH_SOURCE[0]}")"

usage=$(cat <<-END
Draw agCN or psCN for multiple loci in the Parascopy output directory.
Usage:   $(basename "${BASH_SOURCE[0]}") -i <directory> [-o <directory>] [-@ <threads>] -m agcn|pscn -- <agcn/pscn.r arguments>

    -i <dir>, --input <dir>
            Input directory.
    -o <dir>, --output <dir>
            Output directory. Default: input/plots/agcn or input/plots/pscn.
    -@ <int>, --threads <int>
            Use <int> threads [default: 4].
    -r <regex>, --regex <regex>
            Only draw subdirectories that match the regular expression.
    -m agcn|pscn, --mode agcn|pscn
            Draw agCN or psCN [default: agcn].

    All arguments after -- are supplied to agcn.r
END
)

input=""
output=""
threads=4
regex=""
mode="agcn"

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
    -m|--mode)
        mode="$(echo "$2" | tr '[:upper:]' '[:lower:]')"
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

if [[ ${mode} != agcn ]] && [[ ${mode} != pscn ]]; then
    >&2 echo "Unexpected mode ${mode}"
fi

if [[ ${input} = "" ]]; then
    >&2 echo "Missing -i/--input argument."
fi

if [[ ${output} = "" ]]; then
    output=${input}"/plots/${mode}"
fi
mkdir -p "$output"

# ==== Draw plots ====

echo "Drawing copy number:"
echo "    Input:   $input"
echo "    Output:  $output"
echo "    Mode:    $mode"
echo "    Threads: $threads"
if [[ ${regex} != "" ]]; then
    echo "    Regex:   $regex"
fi

export mode="$mode"
export output="$output"
export wdir="$wdir"

find "$input" -type d -name extra -regex ".*$regex.*" | \
    xargs -i -P "$threads" sh -c \
    '${wdir}/${mode}.r -i "{}/.." -o "$output" '"$*"

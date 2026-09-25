SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIN_PYTHON_VERSION="3.9"

print_pyenv_tip() {
    if [ "$(uname -s)" = "Darwin" ] && ! command -v pyenv &> /dev/null
    then
        echo ""
        echo "Dica: o pyenv não está instalado. Ele permite instalar e alternar"
        echo "entre versões do Python de forma isolada, respeitando o arquivo .python-version."
        echo "Para instalar: brew install pyenv"
        echo "Depois: pyenv install \$(cat .python-version) && pyenv local \$(cat .python-version)"
    fi
}

if [ -x "$SCRIPT_DIR/venv/bin/python3" ]
then
    PYTHON_BIN="$SCRIPT_DIR/venv/bin/python3"
elif command -v python3 &> /dev/null
then
    PYTHON_BIN="python3"
elif command -v python &> /dev/null
then
    PYTHON_BIN="python"
else
    echo "Python is not installed"
    exit 1
fi

if ! "$PYTHON_BIN" -c "import sys; sys.exit(0 if sys.version_info >= tuple(map(int, '$MIN_PYTHON_VERSION'.split('.'))) else 1)"
then
    echo "Erro: é necessário Python >= $MIN_PYTHON_VERSION (encontrado: $("$PYTHON_BIN" --version 2>&1) em $PYTHON_BIN)."
    echo "Veja o arquivo .python-version para a versão recomendada."
    print_pyenv_tip
    exit 1
fi

if [ -f "$SCRIPT_DIR/.python-version" ]
then
    EXPECTED_VERSION="$(tr -d '[:space:]' < "$SCRIPT_DIR/.python-version")"
    ACTUAL_VERSION="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    if [ "$ACTUAL_VERSION" != "$EXPECTED_VERSION" ]
    then
        echo "Aviso: .python-version pede Python $EXPECTED_VERSION, mas está sendo usado Python $ACTUAL_VERSION ($PYTHON_BIN)."
        print_pyenv_tip
        echo ""
    fi
fi

"$PYTHON_BIN" src/main.py
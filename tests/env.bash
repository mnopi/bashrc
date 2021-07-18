# shellcheck disable=SC2015

TEST_MACOS="$(uname -a | grep -i darwin 2>/dev/null)"
TEST_LOGNAME="$( test -n "${DARWIN}" && stat -f "%Su" /dev/console || logname )"
TEST_LOGNAMEHOME="$( bash -c "cd ~$( printf %q "${LOGNAME}" ) && pwd" )"
TEST_ROOTHOME=~root
TEST_LOGGEDINUSER="$( echo "show State:/Users/ConsoleUser" | scutil | awk '/Name :/ && ! /loginwindow/ { print $3 }' )"

if test -n "${MACOS}"; then
  TEST_LOGNAMEREALNAME="$( dscl . -read /Users/"${LOGNAME}" RealName RealName | sed -n 's/^ //g;2p' )"
else
  TEST_LOGNAMEREALNAME="$( id -nu )"
fi

TEST_MULTILINE="
First
Second
Last
"

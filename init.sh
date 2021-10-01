#!/bin/bash

shopt -s globstar

# Check script options
if [[ "$#" -lt 1 ]]; then
    echo "Usage: ./init.sh MyAddonName [folder]"
    exit 1
fi

if [[ $(echo "$1" | head -c 1) == "-" ]]; then
    echo "Usage: ./init.sh MyAddonName [folder]"
    exit 1
fi

# Check script necessary tools : sed, perl, perl-rename
PERL=$(command -v perl)
if [[ ! -x "$PERL" ]] ; then
  echo "Install perl"
  exit 1
fi

RENAME=$(command -v perl-rename)
if [[ ! -x "$RENAME" ]] ; then
  RENAME=$(command -v rename)
fi

RENAME_KIND=$($RENAME --help | grep -i PERLEXPR)
if [[ "$RENAME_KIND" == "" ]]; then
  echo "Install perl-rename (sometimes called just 'rename')"
  exit 1
fi

SED=/usr/bin/sed
if [[ "$OSTYPE" == "darwin"* ]]; then
  if ! [[ -x "$(command -v gsed)" ]]; then
    echo "Install gnu-sed"
    exit 1
  fi
  SED=/usr/bin/gsed
fi


ADDON="$1"
ADDON_LC=$(echo $ADDON | $PERL -ne 'print lc')
ADDON_LC_DASHES=$(echo $ADDON_LC | sed 's/_/-/g')

ADDON_DIR="$PWD"

if [[ "x$2" != "x" ]]; then
  TARGET_FOLDER="$2"
  mkdir -p "$TARGET_FOLDER"
  cp -rf * "$TARGET_FOLDER/"
  cd "$TARGET_FOLDER"
else
  TARGET_FOLDER="$PWD"
fi

mv "MyProcess" "$ADDON"
$RENAME "s/my_process/$ADDON_LC/" **/*.{hpp,cpp,txt}
$RENAME "s/MyProcess/$ADDON/" **/*.{hpp,cpp,txt}
$SED -i "s/my_process/$ADDON_LC/g" **/*.{hpp,cpp,txt}
$SED -i "s/MyProcess/$ADDON/g" **/*.{hpp,cpp,txt} release.sh
$SED -i "s/my-process/$ADDON_LC_DASHES/g" **/*.{hpp,cpp,txt} release.sh

echo -e "# $ADDON\nA new and wonderful [ossia score](https://ossia.io) add-on" > README.md


find . -name '*.hpp' -exec $PERL -pi -e 'chomp(my $uidgen = `uuidgen`);s|00000000-0000-0000-0000-000000000000|$uidgen|gi' {} \;
find . -name '*.cpp' -exec $PERL -pi -e 'chomp(my $uidgen = `uuidgen`);s|00000000-0000-0000-0000-000000000000|$uidgen|gi' {} \;
find . -name '*.json' -exec $PERL -pi -e 'chomp(my $uidgen = `uuidgen`);s|00000000-0000-0000-0000-000000000000|$uidgen|gi' {} \;

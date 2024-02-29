#!/bin/bash

# https://github.com/unegma/bash-functions/blob/main/update.sh

# First commit all changes needed to master
# Run this script.

# Run me by using
# ./publish -v patch
# ./publish -v major
# ./publish -v minor
# to increment each of those by just 1.

VERSION=""

#get parameters
while getopts v: flag
do
  case "${flag}" in
    v) VERSION=${OPTARG};;
  esac
done

#get highest tag number, and add 1.0.0 if doesn't exist
CURRENT_VERSION=`git describe --abbrev=0 --tags 2>/dev/null`

if [[ $CURRENT_VERSION == '' ]]
then
  CURRENT_VERSION='1.0.0'
fi
echo "Current Version: $CURRENT_VERSION"

#replace . with space so can split into an array
CURRENT_VERSION=(${CURRENT_VERSION//v/})
CURRENT_VERSION_PARTS=(${CURRENT_VERSION//./ })

#get number parts
VNUM1=${CURRENT_VERSION_PARTS[0]}
VNUM2=${CURRENT_VERSION_PARTS[1]}
VNUM3=${CURRENT_VERSION_PARTS[2]}

if [[ $VERSION == 'major' ]]
then
  VNUM1=$((VNUM1+1))
  VNUM2=0
  VNUM3=0
elif [[ $VERSION == 'minor' ]]
then
  VNUM2=$((VNUM2+1))
  VNUM3=0
elif [[ $VERSION == 'patch' ]]
then
  VNUM3=$((VNUM3+1))
else
  echo "No version type or incorrect type specified, try: -v [major, minor, patch]"
  exit 1
fi

#create new tag
NEW_TAG="$VNUM1.$VNUM2.$VNUM3"
echo "($VERSION) updating $CURRENT_VERSION to $NEW_TAG"

#get current hash and see if it already has a tag
# GIT_COMMIT=`git rev-parse HEAD`
# NEEDS_TAG=`git describe --contains $GIT_COMMIT 2>/dev/null`

echo "Tagged with $NEW_TAG"

# Github stuff
python update_version.py $NEW_TAG
git add setup.py
git commit -m "update version"

git tag $NEW_TAG
git push --tags
git push origin master

# Pypi stuff
python -m build
twine upload --skip-existing dist/*

gh release create $NEW_TAG --notes-from-tag --verify-tag --title $NEW_TAG

exit 0
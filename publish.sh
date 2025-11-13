#!/bin/bash

# https://github.com/unegma/bash-functions/blob/main/update.sh

# First commit all changes needed to master
# Run this script.

# Run me by using
# ./publish -v patch
# ./publish -v minor
# ./publish -v major
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
#Go online and get the latest tag number assoc with a "release"
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
MAJOR=${CURRENT_VERSION_PARTS[0]}
MINOR=${CURRENT_VERSION_PARTS[1]}
PATCH=${CURRENT_VERSION_PARTS[2]}

if [[ $VERSION == 'major' ]]
then
  MAJOR=$((MAJOR+1))
  MINOR=0
  PATCH=0
elif [[ $VERSION == 'minor' ]]
then
  MINOR=$((MINOR+1))
  PATCH=0
elif [[ $VERSION == 'patch' ]]
then
  PATCH=$((PATCH+1))
else
  echo "No version type or incorrect type specified, try: -v [major, minor, patch]"
  exit 1
fi

#create new tag with the new version number
NEW_TAG="$MAJOR.$MINOR.$PATCH"
echo "($VERSION) updating $CURRENT_VERSION to $NEW_TAG"


# Github stuff
python update_version.py $NEW_TAG
git add pyproject.toml
git commit -m "update version"

# cd documentation_source
# make clean
# make html
# make clean
# cd ..
# git add docs/
# git commit -m "docs"

git tag $NEW_TAG
git push --tags
git push origin master develop

gh release create $NEW_TAG --notes-from-tag --verify-tag --title $NEW_TAG

# Pypi
python -m build
twine upload --skip-existing dist/*

# exit 0

# Move old distribution to the old folder.
mv dist/* .dist_old/

# build the package
python3 -m build

# Upload new package
python3 -m twine upload --repository testpypi dist/*
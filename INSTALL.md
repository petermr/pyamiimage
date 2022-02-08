# preamble

COPIED FROM PYAMI AND NOT EDITED.
IGNORE


There are two sections here: for those who want to download and use `py4ami` and those who are developing and uploading.


# Installation for users (download)
`pyamiimage` is now a `PyPI` project and by default will be installed from there. `pyamiimage` is now versioned as `d.d.d` (major/minor/patch). Generally you should install the latest version unless indicated.

## location
See https://pypi.org/project/pyamiimage/

# ====================================
# installation for developers (upload)
# ====================================

## quick instructions for experienced users
```cd pyamiimage
rm -rf dist
```
# edit version number in setup.py





## repeat instructions (ONLY if you've done this before)  ============
```
cd pyamiimage
rm -rf dist
```
# <edit version in setup.py>
```	
python setup.py sdist
twine upload dist/*
````
# <login is pypi, not github>


## for new developers =============

**Follow this carefully. Make sure you are uploading the latest version**

```
cd <pyami> # wherever `pyami` is located
```
This should contain files like:
```
ls -l
total 88
-rw-r--r--@  1 pm286  staff   1502  8 Feb 10:29 setup.py
-rw-r--r--@  1 pm286  staff    194  8 Feb 10:21 requirements.txt
-rw-r--r--@  1 pm286  staff     73  8 Feb 10:18 MANIFEST.in
-rw-r--r--@  1 pm286  staff   5526  8 Feb 10:16 INSTALL.md
drwxr-xr-x  28 pm286  staff    896  6 Feb 22:53 pyimage
drwxr-xr-x  21 pm286  staff    672  6 Feb 22:37 test
drwxr-xr-x   5 pm286  staff    160  5 Feb 14:43 presentations
drwxr-xr-x   3 pm286  staff     96  5 Feb 10:10 logs
-rw-r--r--   1 pm286  staff    470 28 Jan 11:20 Notes.md
-rw-r--r--   1 pm286  staff    876 17 Jan 09:13 README.md
drwxr-xr-x   9 pm286  staff    288  9 Jan 16:38 assets
drwxr-xr-x  24 pm286  staff    768  7 Jan 13:46 temp
drwxr-xr-x   5 pm286  staff    160  7 Dec 10:38 examples
drwxr-xr-x   6 pm286  staff    192 23 Nov 10:18 docs
-rw-r--r--   1 pm286  staff      0 23 Nov 10:18 __init__.py
-rw-r--r--   1 pm286  staff    380 23 Nov 10:18 README.rst
drwxr-xr-x   2 pm286  staff     64 15 Oct 17:11 dist
-rw-r--r--   1 pm286  staff  11357 15 Oct 17:04 LICENSE
````
## edit the version in `setup.py`

**Every upload should have a new increased version, even if the edits are minor.**

Find the version number in `setup.py` and increase it:
````
    name='pyamiimage',
    url='https://github.com/petermr/pyamiimage',
    version='0.0.6',    # increased from `0.0.5`
    description='Image analysis for words and graphics',
    long_description=readme

````
## remove old `dist/`
````
 rm -r dist
```` 
If you don't do this it will upload the previous dist and probably throw errors. 


## create MANIFEST.in

Check MANIFEST.in.
Note that `graft` includes the full subtree. We include the test data which makes the distrib about 20 Mb.
````
include LICENSE
graft py4amiimage/test/resources
````

## create distribution (`dist`)
````
python setup.py sdist
````
This outputs the following 
````

````
## install `twine`
````
pip install twine
````

## upload `dist` to `PyPI`
````
twine upload dist/*
````
gives a login (your `PyPI` login, not github)
````
Uploading distributions to https://upload.pypi.org/legacy/
Enter your username: petermr
Enter your password: 
Uploading py4ami-0.0.6.tar.gz
100%|████████████████████████████████████████████████████████████████████████████████████| 93.7k/93.7k [00:01<00:00, 51.6kB/s]
NOTE: Try --verbose to see response content.

View at:
https://pypi.org/project/pyamiimage/<version>/
````


## release new version

remove old dist
```
rm -rf dist/

```
```
pip install pipreqs --force
pipreqs pyami

```
* cd pyamiimage top directory
* edit version in setup.py
```
 python setup.py bdist_wheel
 twine upload/dist*
```

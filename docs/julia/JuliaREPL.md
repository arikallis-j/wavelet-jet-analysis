# julia-repl

`Ctrl` + `L` - clear session output

## Julia Pkg commands

work with packages:

```bash
# add package
(@ocean) pkg> add StaticArrays
# add definded package version
(@ocean) pkg> add StaticArrays@1.9.16
# add package from github
(@ocean) pkg> add https://github.com/JuliaArrays/StaticArrays.jl
# remove package 
(@ocean) pkg> rm StaticArrays
# update package to the last version
(@ocean) pkg> up StaticArrays

# update all packages to the last version
(@ocean) pkg> up

# check installed packages
(@ocean) pkg> status
Status `~/.../Project.toml`
  [90137ffa] StaticArrays v1.9.18
# check all installed packages
(@ocean) pkg> status --manifes
Status `~/.../Manifest.toml`
  [aea7be01] PrecompileTools v1.3.4
  [21216c6a] Preferences v1.5.2
  [90137ffa] StaticArrays v1.9.18
  ...
```

work with enviroments:

```bash
# activate new enviroment
(@ocean) pkg> activate .
(wavelet-jet-analysis) pkg>

# back to global enviroment
(wavelet-jet-analysis) pkg> activate
(@ocean) pkg>

# load local enviroment dependencies from Project.toml
(wavelet-jet-analysis) pkg> instantiate

# precomile loaded packages
(@ocean) pkg> precompile

# delete unuseable versions of packages
(@ocean) pkg> gc --all
      Active manifest files: 2 found
      Active artifact files: 0 found
      Active scratchspaces: 0 found
     Deleted 1 package installation (1.031 MiB)
     Deleted 1 repo (4.078 MiB) 
```

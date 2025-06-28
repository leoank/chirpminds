default:
  @just --list

dev: studio studioml

[working-directory: 'nix/arion']
studio:
  pueue add 'arion up'

[working-directory: 'nix/arion']
studioml:
  pueue add 'bash studioml.sh'

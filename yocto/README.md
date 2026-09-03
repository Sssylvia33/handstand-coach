# Handstand Coach Yocto learning track

This directory is a small, buildable Yocto exercise attached to Handstand Coach. It is
deliberately separate from the computer-vision implementation: the first milestone is to
understand how an embedded Linux image is assembled, not to cross-build PyTorch.

## Milestone 1 outcome

Build `handstand-learning-image` for `qemux86-64`, boot it with QEMU, and verify that:

- the custom `meta-handstand` layer is enabled;
- BitBake builds and installs the `handstand-hello` recipe;
- the image contains Python;
- systemd starts `handstand-hello.service`; and
- `/run/handstand-hello.status` is created at boot.

The data flow is:

```text
metadata + source files -> BitBake tasks -> packages -> root filesystem -> bootable image
       meta-handstand       do_* tasks    handstand-hello   custom image       QEMU
```

## Why Scarthgap

This exercise targets the Yocto Project 5.0 `scarthgap` branch. It is an LTS series supported
until April 2028 and is a good fit for the available WSL resources. Do not mix layers from
different Yocto release branches.

## Host layout

Build inside the WSL Linux filesystem, not under `/mnt/c` or `/mnt/d`. The generated build tree
contains many small files and relies on normal Linux filesystem behavior.

```text
~/yocto-work/
├── poky/                   # Yocto reference distribution and BitBake
├── meta-handstand/         # copy of this repository's learning layer
└── build-handstand/        # generated; never commit this directory
```

The checked machine has Ubuntu 24.04 under WSL2, about 8 GiB assigned RAM, and ample Linux-side
disk capacity. Keep the first image minimal and limit parallelism.

## 1. Install build-host packages

Open the Ubuntu WSL terminal and run:

```sh
sudo apt update
sudo apt install build-essential chrpath cpio debianutils diffstat file gawk gcc git \
  iputils-ping libacl1 liblz4-tool locales python3 python3-git python3-jinja2 \
  python3-pexpect python3-pip python3-subunit socat texinfo unzip wget xz-utils zstd
sudo locale-gen en_US.UTF-8
```

## 2. Fetch Poky and copy the learning layer

```sh
mkdir -p ~/yocto-work
cd ~/yocto-work
git clone --branch yocto-5.0.19 --single-branch --depth 1 \
  https://git.yoctoproject.org/poky
cp -a /mnt/d/yoga_app/yocto/meta-handstand ~/yocto-work/
```

The release tag makes the exercise reproducible; the `scarthgap` branch continues to receive LTS
updates. Copying the layer is intentional: BitBake stays entirely on the Linux filesystem. Repeat the
copy after editing this repository, or later keep the whole Git repository in WSL.

## 3. Initialize and configure the build

`oe-init-build-env` prepares shell variables and creates the build configuration:

```sh
cd ~/yocto-work/poky
source oe-init-build-env ~/yocto-work/build-handstand
bitbake-layers add-layer ~/yocto-work/meta-handstand
```

Append the contents of this repository's `yocto/handstand.conf` to `conf/local.conf`:

```conf
MACHINE = "qemux86-64"
INIT_MANAGER = "systemd"
BB_NUMBER_THREADS = "4"
PARALLEL_MAKE = "-j 4"
```

Keeping the small fragment in source control records the important local policy while the full,
generated `local.conf` remains outside the repository.

Useful checks before the expensive build:

```sh
bitbake-layers show-layers
bitbake-layers show-recipes handstand-hello
bitbake -e handstand-hello | less
bitbake -g handstand-learning-image
```

The last command generates dependency graph files without building the image.

## 4. Build and boot

```sh
bitbake handstand-learning-image
runqemu qemux86-64 nographic
```

The first build downloads and compiles an entire Linux distribution and can take a long time.
Later builds reuse `downloads`, `sstate-cache`, and prior task output.

At the QEMU login prompt, log in as `root` and inspect the result:

```sh
cat /run/handstand-hello.status
systemctl status handstand-hello --no-pager
handstand-hello
python3 --version
```

Use `Ctrl-A`, then `X`, to exit QEMU in no-graphics mode.

## What each file teaches

| File | Concept |
|---|---|
| `handstand.conf` | Local machine, init-system, and build-resource policy |
| `conf/layer.conf` | How BitBake discovers recipes and checks release compatibility |
| `handstand-hello_1.0.bb` | Recipe metadata, runtime dependencies, install tasks, and systemd integration |
| `handstand-hello.py` | Source payload installed into the target root filesystem |
| `handstand-hello.service` | Target boot behavior |
| `handstand-learning-image.bb` | Image composition through packages |

## Suggested next milestones

1. Change the greeting, rebuild, and identify which tasks rerun.
2. Use `bitbake -c clean handstand-hello` and rebuild only the recipe.
3. Add a configuration file under `/etc/handstand-coach/`.
4. Package the dependency-light session reader from the main application.
5. Add camera/OpenCV support.
6. Evaluate an embedded inference backend; keep Ultralytics/PyTorch out of the base lesson.

## Official references

- [Yocto Project 5.0.19 Quick Build](https://docs.yoctoproject.org/5.0.19/brief-yoctoprojectqs/index.html)
- [Yocto Project concepts](https://docs.yoctoproject.org/5.0.19/overview-manual/concepts.html)
- [Creating and managing layers](https://docs.yoctoproject.org/5.0.19/dev-manual/layers.html)
- [Writing a new recipe](https://docs.yoctoproject.org/5.0.19/dev-manual/new-recipe.html)

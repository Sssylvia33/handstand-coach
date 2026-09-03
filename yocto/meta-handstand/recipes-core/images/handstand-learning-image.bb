SUMMARY = "Minimal QEMU image for learning Yocto with Handstand Coach"

require recipes-core/images/core-image-minimal.bb

IMAGE_INSTALL:append = " handstand-hello"


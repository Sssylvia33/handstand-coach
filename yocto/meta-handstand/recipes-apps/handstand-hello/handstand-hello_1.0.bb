SUMMARY = "Small target-side smoke test for the Handstand Coach learning image"
DESCRIPTION = "Installs a Python command and a systemd oneshot service for the first Yocto exercise."
HOMEPAGE = "https://github.com/Sssylvia33/handstand-coach"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = "file://handstand-hello.py \
           file://handstand-hello.service"

S = "${WORKDIR}"

inherit allarch systemd

RDEPENDS:${PN} = "python3-core"

SYSTEMD_SERVICE:${PN} = "handstand-hello.service"
SYSTEMD_AUTO_ENABLE:${PN} = "enable"

do_install() {
    install -d ${D}${bindir}
    install -m 0755 ${WORKDIR}/handstand-hello.py ${D}${bindir}/handstand-hello

    install -d ${D}${systemd_system_unitdir}
    install -m 0644 ${WORKDIR}/handstand-hello.service \
        ${D}${systemd_system_unitdir}/handstand-hello.service
}

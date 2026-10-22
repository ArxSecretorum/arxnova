// Значок Arx Nova — общий генератор семьи Arx (как у Arx Player для macOS: arxplayer/apps/mac/design/
// make-icon.swift): плитка 824 с полями и тенью, щит-крепость #6C7CE0 с тенью #3A4593 на #14161F,
// всё в sRGB. Внутри — четырёхлучевая звезда «нова», как у первого значка Nova.
import AppKit

let out = CommandLine.arguments[1]
let size = 1024, inset: CGFloat = 100, radius: CGFloat = 185
// Рисуем в sRGB явно: в «устройстве» (deviceRGB) цвета пересчитывались, и башня выходила светлее
// #6C7CE0 (#7F92E6), фон — не #14161F; значки на Mac расходились с Android и с Arx Nova.
let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
let cg = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0, space: srgb,
                   bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
func hex(_ v: UInt32) -> CGColor {
    CGColor(srgbRed: CGFloat(v >> 16 & 0xFF) / 255, green: CGFloat(v >> 8 & 0xFF) / 255, blue: CGFloat(v & 0xFF) / 255, alpha: 1)
}
let tile = CGRect(x: inset, y: inset, width: CGFloat(size) - 2 * inset, height: CGFloat(size) - 2 * inset)
let tilePath = CGPath(roundedRect: tile, cornerWidth: radius, cornerHeight: radius, transform: nil)
cg.saveGState()
cg.setShadow(offset: CGSize(width: 0, height: -8), blur: 20, color: CGColor(srgbRed: 0, green: 0, blue: 0, alpha: 0.35))
cg.setFillColor(hex(0x14161F)); cg.addPath(tilePath); cg.fillPath()
cg.restoreGState()
cg.addPath(tilePath); cg.clip()
// адаптивная иконка Android: видимая часть — центральные 72 из 108 dp → 170.667 из 256
let view0: CGFloat = 42.667, viewSize: CGFloat = 170.667
let k = tile.width / viewSize
cg.translateBy(x: tile.minX, y: tile.maxY)
cg.scaleBy(x: k, y: -k)
cg.translateBy(x: -view0, y: -view0)
cg.translateBy(x: 0, y: 8)
cg.translateBy(x: 128, y: 128); cg.scaleBy(x: 0.9, y: 0.9); cg.translateBy(x: -128, y: -128)

func poly(_ pts: [(CGFloat, CGFloat)], into p: CGMutablePath) {
    p.move(to: CGPoint(x: pts[0].0, y: pts[0].1))
    for (x, y) in pts.dropFirst() { p.addLine(to: CGPoint(x: x, y: y)) }
    p.closeSubpath()
}
func shield(_ dx: CGFloat, _ dy: CGFloat) -> CGMutablePath {
    let p = CGMutablePath()
    poly([(73,56),(95,56),(95,72),(117,72),(117,56),(139,56),(139,72),(161,72),(161,56),(183,56),(183,142),(128,198),(73,142)]
         .map { ($0.0 + dx, $0.1 + dy) }, into: p)
    return p
}
cg.setFillColor(hex(0x3A4593)); cg.addPath(shield(3, 4)); cg.fillPath(using: .evenOdd)
cg.setFillColor(hex(0x6C7CE0)); cg.addPath(shield(0, 0)); cg.fillPath(using: .evenOdd)
// звезда «нова»: 4 луча, центр чуть выше середины щита (координаты значка 256×256)
let star = CGMutablePath()
let cx: CGFloat = 128, cy: CGFloat = 120, R: CGFloat = 40, r: CGFloat = 9
for i in 0..<8 {
    let a = CGFloat.pi / 4 * CGFloat(i) - CGFloat.pi / 2
    let rad = i % 2 == 0 ? R : r
    let pt = CGPoint(x: cx + rad * cos(a), y: cy + rad * sin(a))
    if i == 0 { star.move(to: pt) } else { star.addLine(to: pt) }
}
star.closeSubpath()
cg.setFillColor(hex(0xE7EAF7)); cg.addPath(star); cg.fillPath()
let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: out) as CFURL, "public.png" as CFString, 1, nil)!
CGImageDestinationAddImage(dest, cg.makeImage()!, nil)
CGImageDestinationFinalize(dest)

// swift-tools-version: 6.2
// mlx-ultratex-swift — Swift/MLX port of UltraTex (2K multi-view texturing of an existing mesh) on
// FLUX.2 klein base 9B. WIP spike for AB-T-0489 P2. Gaea-plugin channel only (AB-D-0231): the 9B base and
// the UltraTex LoRAs are under BFL's FLUX Non-Commercial License v2.1.

import PackageDescription

let package = Package(
    name: "UltraTex",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "UltraTex", targets: ["UltraTex"]),
        .executable(name: "ultratex-bench", targets: ["ultratex-bench"]),
        .executable(name: "ultratex-gate", targets: ["ultratex-gate"]),
    ],
    dependencies: [
        // >= 0.32.3: carries the fix for the half-precision split-K GEMM (mlx#3797 / #3810).
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.32.3"),
    ],
    targets: [
        .target(
            name: "UltraTex",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/UltraTex"
        ),
        .executableTarget(
            name: "ultratex-bench",
            dependencies: [
                "UltraTex",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/ultratex-bench"
        ),
        .executableTarget(
            name: "ultratex-gate",
            dependencies: [
                "UltraTex",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/ultratex-gate"
        ),
    ]
)

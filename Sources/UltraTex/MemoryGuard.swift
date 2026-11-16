// A hard stop on this process's MLX memory, and a check that the machine has room before a run starts.
//
// Why this exists: on 2026-11-15 an untiled FLUX.2 autoencoder pass at 2048² (a 17 GB attention score matrix per
// encode, about 42 GB per decode) ran here with MLX's default buffer cache, which keeps freed buffers up to the
// memory limit. Eight such calls in one process, on a Mac other sessions were also using, hung the system.
// MLX's own memory limit is a guideline, not a cap: it keeps allocating while the system can page. So the cap is
// enforced here: a small cache limit, and a watchdog that ends the process when active + cached MLX memory passes
// the budget. Dying early is the point.

import Darwin
import Foundation
import MLX

public enum UltraTexMemory {
    /// Bytes the system could hand out now without paging: free, inactive, speculative and purgeable pages.
    public static func availableBytes() -> Int {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        let pages = Int(stats.free_count) + Int(stats.inactive_count) + Int(stats.speculative_count)
            + Int(stats.purgeable_count)
        return pages * Int(sysconf(_SC_PAGESIZE))
    }

    /// The largest physical footprint this process has had, in bytes (what Activity Monitor calls Memory, at its
    /// peak): the figure a lane is charged by.
    public static func peakFootprintBytes() -> Int {
        var info = rusage_info_v4()
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0) }
        }
        return result == 0 ? Int(info.ri_lifetime_max_phys_footprint) : 0
    }

    /// Call once, before any model is built. Refuses to start (exit 75) unless the machine has `capGB` plus a
    /// 16 GB margin available, bounds MLX's buffer cache, and starts the watchdog. Returns the available GB seen.
    @discardableResult
    public static func guardProcess(capGB: Double, cacheGB: Double = 8, label: String = "ultratex") -> Double {
        let available = Double(availableBytes()) / 1_073_741_824
        let margin = 16.0
        if available < capGB + margin {
            FileHandle.standardError.write(
                "\(label): refusing to start. The run may use up to \(Int(capGB)) GB of MLX memory and wants \(Int(margin)) GB of margin, but only \(String(format: "%.0f", available)) GB is available. Free memory or pass a smaller --max-gb.\n"
                    .data(using: .utf8)!)
            exit(75)
        }
        MLX.Memory.cacheLimit = Int(cacheGB * 1_073_741_824)
        let cap = Int(capGB * 1_073_741_824)
        let thread = Thread {
            while true {
                let used = MLX.Memory.activeMemory + MLX.Memory.cacheMemory
                if used > cap {
                    let text = "\(label): MLX memory reached \(used >> 30) GB, over the \(Int(capGB)) GB cap. Stopping this process to protect the machine.\n"
                    FileHandle.standardError.write(text.data(using: .utf8)!)
                    _exit(75)
                }
                usleep(200_000)
            }
        }
        thread.qualityOfService = .userInteractive
        thread.start()
        return available
    }
}

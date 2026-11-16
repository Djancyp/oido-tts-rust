//! Embeds the llama.cpp library locations in the binary's rpath: the dev build directory
//! reported by `llama-sys`, then `lib/` next to the executable for packaged installs.

fn main() {
    println!("cargo:rerun-if-env-changed=DEP_LLAMA_LIB_DIR");
    if let Ok(dir) = std::env::var("DEP_LLAMA_LIB_DIR") {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{dir}");
    }
    println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN/lib");
}

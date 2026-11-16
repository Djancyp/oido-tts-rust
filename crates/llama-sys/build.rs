//! Generates bindings against a prebuilt llama.cpp checkout.
//!
//! `LLAMA_CPP_DIR` must point at a llama.cpp source tree whose shared libraries were built
//! with CMake (`BUILD_SHARED_LIBS=ON`, the default). `LLAMA_LIB_DIR` overrides where the
//! `.so` files live (default: `$LLAMA_CPP_DIR/build/bin`). That directory is exported to
//! direct dependents' build scripts as `DEP_LLAMA_LIB_DIR` so binaries can embed it in rpath.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_LIB_DIR");
    println!("cargo:rerun-if-changed=wrapper.h");

    let src = PathBuf::from(env::var("LLAMA_CPP_DIR").unwrap_or_else(|_| {
        panic!("LLAMA_CPP_DIR is not set: point it at a llama.cpp checkout built with CMake")
    }));
    let lib_dir = env::var("LLAMA_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| src.join("build").join("bin"));
    assert!(
        lib_dir.join("libmtmd.so").exists() || lib_dir.join("libmtmd.so.0").exists(),
        "libmtmd not found in {}: build llama.cpp or set LLAMA_LIB_DIR",
        lib_dir.display()
    );

    let includes = [
        src.join("include"),
        src.join("ggml").join("include"),
        src.join("tools").join("mtmd"),
    ];
    for dir in &includes {
        println!("cargo:rerun-if-changed={}", dir.display());
    }

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_args(includes.iter().map(|d| format!("-I{}", d.display())))
        .allowlist_function("(llama|mtmd)_.*")
        .allowlist_type("(llama|mtmd)_.*")
        .allowlist_var("(LLAMA|MTMD)_.*")
        .prepend_enum_name(false)
        .derive_default(true)
        .generate()
        .expect("bindgen failed on llama.cpp headers");
    let out = PathBuf::from(env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("writing bindings.rs");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    for lib in ["mtmd", "llama", "ggml", "ggml-base"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    println!("cargo:lib_dir={}", lib_dir.display());
}

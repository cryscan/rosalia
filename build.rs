use std::path::Path;
use std::process::Command;

// ============================================================
// Types
// ============================================================

/// A single variant of a shader preprocessor define.
struct DefineVariant {
    /// How to emit the -D flag:
    ///   None       → no flag
    ///   Some("")   → flag only (-DNAME)
    ///   Some("v")  → value define (-DNAME=v)
    shader_value: Option<String>,
    /// Component to include in the output filename. Empty = omitted.
    filename_value: &'static str,
}

/// A dimension of shader preprocessor defines (e.g., A_TYPE, ACTIVATION).
struct DefineDimension {
    name: &'static str,
    variants: Vec<DefineVariant>,
}

/// A shader specification with its source file and define dimensions.
struct ShaderSpec {
    name: &'static str,
    source: &'static str,
    defines: Vec<DefineDimension>,
}

// ============================================================
// Variant constructors
// ============================================================

/// Value define: `-DNAME=value`, with a filename component.
///
/// Use an empty `filename` to omit this dimension from the output filename.
fn def<T: std::fmt::Display>(value: T, filename: &'static str) -> DefineVariant {
    DefineVariant {
        shader_value: Some(value.to_string()),
        filename_value: filename,
    }
}

/// Flag define: `-DNAME` (no value), with a filename component.
fn flag(filename: &'static str) -> DefineVariant {
    DefineVariant {
        shader_value: Some(String::new()),
        filename_value: filename,
    }
}

/// No define: no `-D` flag emitted, with a filename component.
fn none(filename: &'static str) -> DefineVariant {
    DefineVariant {
        shader_value: None,
        filename_value: filename,
    }
}

// ============================================================
// Macro for declaring shader compilation specs
// ============================================================

/// Declare shader variants for compilation.
///
/// Usage:
/// ```ignore
/// compile_shaders! {
///     "name" => "src/shader.comp" => {
///         DEFINE_A => [def("value1", "fn1"), def("value2", "fn2")],
///         DEFINE_B => [none("default"), flag("enabled")],
///         DEFINE_C => [none("off"), def(1, "on")],
///     },
/// }
/// ```
///
/// Variant constructors:
/// - `def(value, filename)` — emits `-DDEFINE=value`; `filename` appears in output name
/// - `flag(filename)`       — emits `-DDEFINE` (no value); `filename` appears in output name
/// - `none(filename)`       — emits no `-D` flag; `filename` appears in output name
///
/// Pass an empty string as `filename` to omit that dimension from the output filename.
macro_rules! compile_shaders {
    (
        $(
            $name:literal => $source:literal => {
                $( $define:ident => [ $($variant:expr),+ $(,)? ] ),+ $(,)?
            }
        ),+ $(,)?
    ) => {
        vec![
            $(
                ShaderSpec {
                    name: $name,
                    source: $source,
                    defines: vec![
                        $(
                            DefineDimension {
                                name: stringify!($define),
                                variants: vec![$($variant),+],
                            },
                        )+
                    ],
                },
            )+
        ]
    };
}

// ============================================================
// Compilation logic
// ============================================================

fn cartesian_product(defines: &[DefineDimension]) -> Vec<Vec<(&str, &DefineVariant)>> {
    let mut result: Vec<Vec<(&str, &DefineVariant)>> = vec![vec![]];
    for dim in defines {
        let mut next = Vec::new();
        for existing in &result {
            for variant in &dim.variants {
                let mut combo = existing.clone();
                combo.push((dim.name, variant));
                next.push(combo);
            }
        }
        result = next;
    }
    result
}

fn compile_shader(shaders_dir: &Path, spv_dir: &Path, spec: &ShaderSpec) {
    let source_path = shaders_dir.join(spec.source);
    println!("cargo:rerun-if-changed={}", source_path.display());

    for combo in cartesian_product(&spec.defines) {
        let filename_parts: Vec<&str> = combo
            .iter()
            .filter_map(|(_, v)| {
                if v.filename_value.is_empty() {
                    None
                } else {
                    Some(v.filename_value)
                }
            })
            .collect();

        let filename = if filename_parts.is_empty() {
            format!("{}.spv", spec.name)
        } else {
            format!("{}_{}.spv", spec.name, filename_parts.join("_"))
        };

        let output_path = spv_dir.join(&filename);

        let mut cmd = Command::new("glslangValidator");
        cmd.arg("--target-env").arg("spirv1.3");

        for (define_name, variant) in &combo {
            match &variant.shader_value {
                None => {}
                Some(v) if v.is_empty() => {
                    cmd.arg(format!("-D{}", define_name));
                }
                Some(v) => {
                    cmd.arg(format!("-D{}={}", define_name, v));
                }
            }
        }

        cmd.arg("-V").arg(&source_path).arg("-o").arg(&output_path);

        let status = cmd
            .status()
            .expect("failed to run glslangValidator — make sure it is installed and in PATH");
        if !status.success() {
            panic!("glslangValidator failed for {}", filename);
        }
    }
}

// ============================================================
// Main
// ============================================================

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let shaders_dir = Path::new("assets/shaders");
    let spv_dir = shaders_dir.join("spv");
    std::fs::create_dir_all(&spv_dir).unwrap();

    let specs = compile_shaders! {
        "gemm" => "src/gemm.comp" => {
            A_BITS => [def(16, "")],
            A_TYPE => [def("float16_t", "f16"), def("float32_t", "f32")],
            C_TYPE => [def("float16_t", "f16"), def("float32_t", "f32")],
            AFFINE => [none(""), flag("affine")],
            ACTIVATION => [none(""), def(1, "relu2"), def(2, "tanh")],
        },
        "norm" => "src/norm.comp" => {
            I_TYPE => [def("float16_t", "f16"), def("float32_t", "f32")],
            O_TYPE => [def("float16_t", "f16"), def("float32_t", "f32")],
            AFFINE => [none(""), flag("affine")],
        },
        "gemv" => "src/gemv.comp" => {
            I_TYPE => [def("float16_t", "f16"), def("float32_t", "f32")],
            O_TYPE => [def("float16_t", "f16"), def("float32_t", "f32")],
            AFFINE => [none(""), flag("affine")],
            ACTIVATION => [none(""), def(1, "relu2"), def(2, "tanh")],
        },
        "token_shift" => "src/token_shift.comp" => {
            I_TYPE => [def("float16_t", "f16"), def("float32_t", "f32")],
            O_TYPE => [def("float16_t", "f16"), def("float32_t", "f32")],
            REVERSED => [none(""), flag("rev")],
        },
    };

    for spec in &specs {
        compile_shader(shaders_dir, &spv_dir, spec);
    }
}

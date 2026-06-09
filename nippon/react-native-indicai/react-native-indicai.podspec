require "json"
pkg = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "react-native-indicai"
  s.version      = pkg["version"]
  s.summary      = pkg["description"]
  s.license      = pkg["license"]
  s.author       = { "IndicAI" => "sdk@indicai.in" }
  s.homepage     = "https://github.com/yourorg/react-native-indicai"
  s.platform     = :ios, "15.0"
  s.source       = { :path => "." }

  # All Swift + ObjC bridge files in ios/ (no separate XCFramework; see README)
  s.source_files = "ios/**/*.{swift,m,h}"

  # Tokenizer vocab / dict / SentencePiece assets bundled with the pod
  s.resources = ["ios/assets/**/*"]

  # ONNX Runtime from CocoaPods — pinned to the same 1.26 generation as the
  # Android side (onnxruntime-android:1.26.0) to keep model/op behavior in parity.
  # iOS has no 16 KB page requirement (Apple arm64 has used 16 KB pages for years).
  s.dependency "onnxruntime-objc", "~> 1.26"

  s.dependency "React-Core"

  # libz is included in the iOS SDK — needed for in-process ZIP extraction
  s.libraries = ["z"]

  s.swift_version = "5.9"

  s.pod_target_xcconfig = {
    "DEFINES_MODULE"              => "YES",
    "SWIFT_OBJC_BRIDGING_HEADER"  => "$(PODS_TARGET_SRCROOT)/ios/IndicAIModule-Bridging-Header.h",
  }
end

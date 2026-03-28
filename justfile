DATETIME    := `date '+%Y-%m-%dT%H%M'`
SHA         := `git rev-parse HEAD`
SHA_TAG     := "sha-" + SHA
GITHUB_REPO := `git remote get-url origin | sed 's|.*github.com[:/]||;s|\.git$||' | tr '[:upper:]' '[:lower:]'`


default:
  just --list


build_image:
  docker buildx create --name unsecure_builder --buildkitd-flags '--allow-insecure-entitlement security.insecure' --use
  docker buildx inspect --bootstrap
  docker buildx build --allow security.insecure -t vllm-rocm-gfx1201:{{DATETIME}}-{{SHA_TAG}} . --load


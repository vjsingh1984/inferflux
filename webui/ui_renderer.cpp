#include "webui/ui_renderer.h"

#include <litehtml.h>

#include "webui/ui_bundle.h"

namespace inferflux {

std::string WebUiRenderer::RenderIndex(const std::string &backend_label) const {
  std::string html = UiHtml();
  litehtml::replace_placeholder(html, "{{css}}", UiCss());
  litehtml::replace_placeholder(html, "{{js}}", UiJs());
  litehtml::replace_placeholder(html, "{{backend}}", backend_label);
  return html;
}

} // namespace inferflux

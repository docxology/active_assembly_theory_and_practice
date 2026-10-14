-- Add only the source-configured DOCX cover, after scientific numbering.
function Pandoc(doc)
  if FORMAT ~= "docx" then return doc end
  -- A relative ../figures path can resolve to committed snapshots through
  -- Pandoc's resource search. Bind manuscript images to canonical outputs.
  local root = pandoc.utils.stringify(doc.meta["assembly-figure-root"] or "")
  if root ~= "" then
    doc = doc:walk({Image = function(image)
      local name = image.src:match("^%.%./figures/([^/]+%.png)$")
      if name then image.src = root .. "/" .. name end
      return image
    end})
  end
  local path = pandoc.utils.stringify(doc.meta["assembly-cover-path"] or "")
  local alt = pandoc.utils.stringify(doc.meta["assembly-cover-alt"] or "")
  if path == "" or alt == "" then error("configured DOCX cover metadata is missing") end
  local image = pandoc.Image(alt, path, "", pandoc.Attr("", {}, {{"width", "3.8in"}}))
  -- Preserve the native Office before/after image-order contract. The
  -- source-owned cover is prepended to both the image and alternative lists.
  local alternatives = doc.meta["template-office-source-alternatives"]
  if alternatives then table.insert(alternatives, 1, pandoc.MetaString(alt)) end
  table.insert(doc.blocks, 1, pandoc.Para({image}))
  table.insert(doc.blocks, 2, pandoc.RawBlock("openxml", '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'))
  return doc
end

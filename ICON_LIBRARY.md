# Portfolio Dashboard Icon Library

## Custom SVG Icon System

I've created a comprehensive custom icon system using inline SVG icons that perfectly match your modern design aesthetic. All icons are scalable, customizable, and optimized for performance.

## ✨ Features

- **24 Custom SVG Icons** - Professional, clean, and modern
- **Fully Scalable** - Perfect at any size (vector graphics)
- **Color Customizable** - Change stroke color via CSS
- **No External Dependencies** - All icons embedded in HTML
- **Fast Loading** - No external icon library downloads
- **Consistent Design** - All icons follow the same design language

## 🎨 Icon Sizes

```html
<svg class="icon icon-sm">   <!-- 16px -->
<svg class="icon icon-md">   <!-- 20px -->
<svg class="icon icon-lg">   <!-- 24px -->
<svg class="icon icon-xl">   <!-- 32px -->
<svg class="icon icon-2xl">  <!-- 48px -->
```

## 📚 Available Icons

### Financial & Portfolio Icons

**Portfolio Icon** - `#icon-portfolio`
```html
<svg class="icon icon-md">
    <use href="#icon-portfolio"/>
</svg>
```

**Dollar Sign** - `#icon-dollar`
```html
<svg class="icon icon-md">
    <use href="#icon-dollar"/>
</svg>
```

**Target** - `#icon-target`
```html
<svg class="icon icon-md">
    <use href="#icon-target"/>
</svg>
```

**Shield/Risk** - `#icon-shield`
```html
<svg class="icon icon-md">
    <use href="#icon-shield"/>
</svg>
```

### Chart & Analytics Icons

**Chart/Bar Chart** - `#icon-chart`
```html
<svg class="icon icon-md">
    <use href="#icon-chart"/>
</svg>
```

**Trending Up** - `#icon-trending-up`
```html
<svg class="icon icon-md">
    <use href="#icon-trending-up"/>
</svg>
```

**Trending Down** - `#icon-trending-down`
```html
<svg class="icon icon-md">
    <use href="#icon-trending-down"/>
</svg>
```

**Pie Chart** - `#icon-pie-chart`
```html
<svg class="icon icon-md">
    <use href="#icon-pie-chart"/>
</svg>
```

**Activity/Pulse** - `#icon-activity`
```html
<svg class="icon icon-md">
    <use href="#icon-activity"/>
</svg>
```

### Action Icons

**Refresh** - `#icon-refresh`
```html
<svg class="icon icon-md">
    <use href="#icon-refresh"/>
</svg>
```

**Check/Success** - `#icon-check`
```html
<svg class="icon icon-md">
    <use href="#icon-check"/>
</svg>
```

**X/Close** - `#icon-x`
```html
<svg class="icon icon-md">
    <use href="#icon-x"/>
</svg>
```

**Filter** - `#icon-filter`
```html
<svg class="icon icon-md">
    <use href="#icon-filter"/>
</svg>
```

**Settings** - `#icon-settings`
```html
<svg class="icon icon-md">
    <use href="#icon-settings"/>
</svg>
```

### Information Icons

**Info** - `#icon-info`
```html
<svg class="icon icon-md">
    <use href="#icon-info"/>
</svg>
```

**Alert/Warning** - `#icon-alert`
```html
<svg class="icon icon-md">
    <use href="#icon-alert"/>
</svg>
```

**Lightbulb/Idea** - `#icon-lightbulb`
```html
<svg class="icon icon-md">
    <use href="#icon-lightbulb"/>
</svg>
```

### Content Icons

**News** - `#icon-news`
```html
<svg class="icon icon-md">
    <use href="#icon-news"/>
</svg>
```

**Social/Twitter** - `#icon-social`
```html
<svg class="icon icon-md">
    <use href="#icon-social"/>
</svg>
```

**Calendar** - `#icon-calendar`
```html
<svg class="icon icon-md">
    <use href="#icon-calendar"/>
</svg>
```

### Special Icons

**Options** - `#icon-options`
```html
<svg class="icon icon-md">
    <use href="#icon-options"/>
</svg>
```

**Zap/Lightning** - `#icon-zap`
```html
<svg class="icon icon-md">
    <use href="#icon-zap"/>
</svg>
```

## 🎯 Usage Examples

### Basic Icon
```html
<svg class="icon icon-md">
    <use href="#icon-chart"/>
</svg>
```

### Icon with Custom Color
```html
<svg class="icon icon-md" style="stroke: var(--accent-gold);">
    <use href="#icon-dollar"/>
</svg>
```

### Icon in Button
```html
<button class="btn icon-button">
    <svg class="icon icon-md">
        <use href="#icon-refresh"/>
    </svg>
    Refresh Data
</button>
```

### Icon in Heading
```html
<h3>
    <svg class="icon icon-md icon-inline" style="stroke: var(--accent-emerald);">
        <use href="#icon-trending-up"/>
    </svg>
    Stock Holdings
</h3>
```

### Icon in List
```html
<li>
    <svg class="icon icon-sm icon-inline">
        <use href="#icon-check"/>
    </svg>
    Feature description
</li>
```

## 🎨 Color Customization

You can customize icon colors using CSS:

```html
<!-- Use theme colors -->
<svg class="icon" style="stroke: var(--accent-gold);">
    <use href="#icon-dollar"/>
</svg>

<!-- Use specific colors -->
<svg class="icon" style="stroke: #10B981;">
    <use href="#icon-trending-up"/>
</svg>

<!-- Red for warnings -->
<svg class="icon" style="stroke: var(--accent-red);">
    <use href="#icon-alert"/>
</svg>
```

## 📊 Already Implemented

Icons have been added to:
- ✅ Tab Navigation (Portfolio, Risk, Options, Analysis, Social)
- ✅ Refresh Button
- ✅ Section Headings (Stock Holdings, Options Holdings, Risk Metrics)
- ✅ Placeholder Content (Stock Analysis, Social Media)
- ✅ Feature Lists

## 🚀 Future Additions

You can easily add more icons by following the same pattern in the SVG sprite:

```html
<symbol id="icon-name" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <!-- SVG path data here -->
</symbol>
```

## 💡 Tips

1. **Consistent Sizing** - Use predefined size classes (icon-sm, icon-md, etc.) for consistency
2. **Semantic Colors** - Use theme colors for consistent branding
3. **Inline Usage** - Use `icon-inline` class for icons within text
4. **Button Icons** - Use `icon-button` class for buttons with icons
5. **Performance** - All icons share the same SVG definition, very efficient!

---

**Created with ❤️ for Gold Pressed Latinum Portfolio Dashboard**


window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, context) {
            const {
                counts,
                scale,
                min,
                max
            } = context.hideout;
            const name = feature.properties.name;
            const count = counts[name] || 0;
            const log_val = Math.log1p(count);
            let norm = (log_val - min) / (max - min);
            norm = Math.max(0, Math.min(1, norm));
            const idx = Math.floor(norm * (scale.length - 1));
            const fillColor = scale[idx] || 'rgba(0,0,0,0)';
            return {
                fillColor: fillColor,
                fillOpacity: 0.7,
                color: feature.properties.style?.color || '#444444',
                weight: 2
            };
        }

    }
});
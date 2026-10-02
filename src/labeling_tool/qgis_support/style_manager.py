from qgis.core import (
    Qgis,
    QgsRendererCategory, QgsCategorizedSymbolRenderer,
    QgsSimpleFillSymbolLayer, QgsFillSymbol,
    QgsSingleSymbolRenderer,
    QgsSingleBandPseudoColorRenderer, QgsColorRampShader, QgsRasterShader,
    QgsPalettedRasterRenderer,
)
from qgis.PyQt.QtGui import QColor


class StyleManager:
    # Colors sourced from 610826 绥德县 QSDK reference (TDLYDM/TDLYMC)
    CLASS_COLORS = {
        12: ("水浇地", "#FFFF00"),
        13: ("旱地", "#E6B43C"),
        21: ("果园", "#FF8000"),
        31: ("有林地", "#007800"),
        32: ("灌木林地", "#50AA3C"),
        33: ("其他林地", "#78C878"),
        43: ("其他草地", "#AADC50"),
        51: ("城镇建设用地", "#DC0000"),
        52: ("农村建设用地", "#FF5050"),
        53: ("人为扰动用地", "#B400B4"),
        54: ("其他建设用地", "#FF78C8"),
        61: ("农村道路", "#505050"),
        62: ("其他交通用地", "#000000"),
        71: ("河湖库塘", "#0064FF"),
    }

    @classmethod
    def apply_categorized_style(cls, layer):
        categories = []
        for code, (name, hex_color) in sorted(cls.CLASS_COLORS.items()):
            color = QColor(hex_color)
            symbol = QgsFillSymbol.createSimple({
                "color": hex_color,
                "color_border": "#232323",
                "width_border": "0.15",
                "style": "solid",
                "style_border": "solid",
            })
            symbol.setOpacity(0.6)
            cat = QgsRendererCategory(code, symbol, name)
            categories.append(cat)

        renderer = QgsCategorizedSymbolRenderer("class_code", categories)
        layer.setRenderer(renderer)

    @classmethod
    def apply_outline_style(cls, layer, color="#66AAFF", width=0.5):
        fill_symbol = QgsFillSymbol.createSimple({
            "color": "#00000000",
            "color_border": color,
            "width_border": str(width),
            "style": "no",
            "style_border": "solid",
        })
        layer.setRenderer(QgsSingleSymbolRenderer(fill_symbol))

    @classmethod
    def apply_accepted_style(cls, layer):
        fill_symbol = QgsFillSymbol.createSimple({
            "color": "#2ECC40",
            "color_border": "#1a1a1a",
            "width_border": "0.2",
            "style": "solid",
            "style_border": "solid",
        })
        fill_symbol.setOpacity(0.4)
        layer.setRenderer(QgsSingleSymbolRenderer(fill_symbol))

    @classmethod
    def apply_confidence_style(cls, layer):
        shader = QgsColorRampShader()
        shader.setColorRampType(Qgis.ShaderInterpolationMethod.Linear)
        items = [
            QgsColorRampShader.ColorRampItem(0.0, QColor("#FF0000"), "0.00"),
            QgsColorRampShader.ColorRampItem(0.25, QColor("#FF6666"), "0.25"),
            QgsColorRampShader.ColorRampItem(0.5, QColor("#FFFF00"), "0.50"),
            QgsColorRampShader.ColorRampItem(0.75, QColor("#66FF66"), "0.75"),
            QgsColorRampShader.ColorRampItem(1.0, QColor("#00FF00"), "1.00"),
        ]
        shader.setColorRampItemList(items)

        raster_shader = QgsRasterShader()
        raster_shader.setRasterShaderFunction(shader)

        renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, raster_shader)
        renderer.setClassificationMin(0.0)
        renderer.setClassificationMax(1.0)

        layer.setRenderer(renderer)

    @classmethod
    def apply_semantic_raster_style(cls, layer):
        """Render index-valued masks while labeling classes with QSDK codes."""
        classes = [
            QgsPalettedRasterRenderer.Class(
                index,
                QColor(hex_color),
                f"{code} {name}",
            )
            for index, (code, (name, hex_color)) in enumerate(cls.CLASS_COLORS.items())
        ]
        layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(), 1, classes))

    @classmethod
    def get_class_name(cls, code):
        return cls.CLASS_COLORS.get(code, ("未知", ""))[0]

    @classmethod
    def get_class_color(cls, code):
        return cls.CLASS_COLORS.get(code, ("", "#000000"))[1]

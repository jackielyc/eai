"""ui.builders.model tests — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_qwen_local = module_ref('eai_app.adapters.models.qwen_local')
_eai_adapters_models_qwen_remote = module_ref('eai_app.adapters.models.qwen_remote')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_images = module_ref('eai_app.ui.widgets.images')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')

def build(self, build):
    build.test_tab = _eai_platform_bootstrap.QWidget()
    test_outer = _eai_platform_bootstrap.QVBoxLayout(build.test_tab)
    test_outer.setContentsMargins(8, 6, 8, 6)
    test_outer.setSpacing(6)

    test_hint = _eai_platform_bootstrap.QLabel(
        "测试区：点击下方场景图，即可附带到右侧「AI 对话」发送"
    )
    test_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};")
    test_hint.setWordWrap(True)
    test_hint.setToolTip(f"图片目录: {_eai_configuration_policies.TEST_IMAGES_DIR}")
    test_outer.addWidget(test_hint)

    test_grid_host = _eai_platform_bootstrap.QWidget()
    test_grid = _eai_platform_bootstrap.QGridLayout(test_grid_host)
    test_grid.setContentsMargins(0, 0, 0, 0)
    test_grid.setHorizontalSpacing(8)
    test_grid.setVerticalSpacing(8)
    self._test_image_labels: _eai_platform_bootstrap.Dict[str, _eai_ui_widgets_images.ScaledPixmapLabel] = {}
    self._test_image_paths: _eai_platform_bootstrap.Dict[str, str] = {}
    self._test_selected_scenario: str = ""
    cols = 4
    for build.idx, build.title in enumerate(_eai_adapters_models_qwen_local.TEST_SCENARIO_LABELS):
        cell = _eai_platform_bootstrap.QWidget()
        cell_layout = _eai_platform_bootstrap.QVBoxLayout(cell)
        cell_layout.setContentsMargins(0, 0, 0, 0)
        cell_layout.setSpacing(4)
        title_label = _eai_platform_bootstrap.QLabel(build.title)
        title_label.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
        title_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL, _eai_platform_bootstrap.QFont.Bold))
        title_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_PRIMARY};")
        title_label.setWordWrap(True)
        cell_layout.addWidget(title_label)
        image_label = _eai_ui_widgets_images.ScaledPixmapLabel("暂无图像")
        image_path = _eai_ui_widgets_images.resolve_test_scenario_image_path(build.title)
        if image_path is not None:
            pix = _eai_platform_bootstrap.QPixmap(image_path)
            if not pix.isNull():
                image_label.set_source_pixmap(pix)
                image_label.set_clickable(True)
                image_label.setToolTip(
                    f"点击选中「{build.title}」用于对话\n{image_path}"
                )
                self._test_image_paths[build.title] = image_path
                image_label.clicked.connect(
                    lambda t=build.title: self._on_test_scenario_clicked(t)
                )
            else:
                image_label.setText("加载失败")
                image_label.setToolTip(f"{build.title}\n无法读取: {image_path}")
        else:
            expected = _eai_adapters_models_qwen_local.TEST_SCENARIO_IMAGE_FILES.get(build.title, f"{build.title}.png")
            image_label.setText("暂无图像")
            image_label.setToolTip(
                f"{build.title}\n未找到: {_eai_platform_bootstrap.os.path.join(_eai_configuration_policies.TEST_IMAGES_DIR, expected)}"
            )
        cell_layout.addWidget(image_label, 1)
        self._test_image_labels[build.title] = image_label
        test_grid.addWidget(cell, build.idx // cols, build.idx % cols)
    for c in range(cols):
        test_grid.setColumnStretch(c, 1)
    for r in range((len(_eai_adapters_models_qwen_local.TEST_SCENARIO_LABELS) + cols - 1) // cols):
        test_grid.setRowStretch(r, 1)
    test_outer.addWidget(test_grid_host, 1)

    service_row = _eai_platform_bootstrap.QHBoxLayout()
    service_row.setSpacing(6)
    service_row.addWidget(_eai_platform_bootstrap.QLabel("部署位置"))
    self.test_qwen_target_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.test_qwen_target_combo.addItem("本地本机", "local")
    for host_id, host_label in _eai_configuration_policies.REMOTE_QWEN_HOSTS:
        self.test_qwen_target_combo.addItem(host_label, f"remote:{host_id}")
    tip_lines = ["本地：本机 GPU 部署。"]
    for host_id, host_label in _eai_configuration_policies.REMOTE_QWEN_HOSTS:
        prof = _eai_adapters_models_qwen_remote.remote_qwen_profile(host_id)
        tip_lines.append(
            f"{host_label}：SSH {host_id}，隧道 "
            f"{prof.get('api_base') or _eai_adapters_models_qwen_remote.remote_qwen_api_base_for_host(host_id)}"
        )
    tip_lines.append("基座与 Lake LoRA 等本地可部署模型均可选任意远程机。")
    self.test_qwen_target_combo.setToolTip("\n".join(tip_lines))
    service_row.addWidget(self.test_qwen_target_combo)
    service_row.addWidget(_eai_platform_bootstrap.QLabel("权重目录"))
    self.test_qwen_root_combo = _eai_ui_widgets_inputs.ImeSafeComboBox()
    self.test_qwen_root_combo.setMinimumWidth(140)
    service_row.addWidget(self.test_qwen_root_combo)
    self.test_qwen_refresh_btn = _eai_platform_bootstrap.QPushButton("刷新")
    self.test_qwen_refresh_btn.setToolTip(
        "扫描所选根目录下含 config.json / adapter_config.json 的子目录"
    )
    service_row.addWidget(self.test_qwen_refresh_btn)
    service_row.addWidget(_eai_platform_bootstrap.QLabel("模型"))
    self.test_qwen_model_combo = _eai_ui_widgets_inputs.FilterableImeSafeComboBox()
    self.test_qwen_model_combo.setMinimumWidth(200)
    self.test_qwen_model_combo.setToolTip(
        "从上方根目录扫描到的可部署权重；点击后可输入关键词模糊过滤，"
        "选中后点「启动推理服务」。"
    )
    service_row.addWidget(self.test_qwen_model_combo)
    self.test_model_path_label = _eai_platform_bootstrap.QLabel("")
    self.test_model_path_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.test_model_path_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};")
    self.test_model_path_label.setWordWrap(True)
    service_row.addWidget(self.test_model_path_label, 1)
    self.test_qwen_status_label = _eai_platform_bootstrap.QLabel("服务: --")
    self.test_qwen_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    service_row.addWidget(self.test_qwen_status_label)
    self.test_qwen_load_last_btn = _eai_platform_bootstrap.QPushButton("上次配置")
    self.test_qwen_load_last_btn.setFixedWidth(64)
    self.test_qwen_load_last_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.test_qwen_load_last_btn.setToolTip(
        "恢复上一次「启动推理服务」时使用的部署位置、权重目录与模型"
    )
    service_row.addWidget(self.test_qwen_load_last_btn)
    self.test_qwen_start_btn = _eai_platform_bootstrap.QPushButton("启动推理服务")
    self.test_qwen_start_btn.setToolTip(
        "按「部署位置」启动本地或远程 OpenAI 兼容推理服务。\n"
        "远程会自动 SSH 同步脚本、拉起模型，并建立本地隧道。\n"
        "首次加载可能需要数分钟。"
    )
    service_row.addWidget(self.test_qwen_start_btn)
    self.test_qwen_stop_btn = _eai_platform_bootstrap.QPushButton("停止")
    self.test_qwen_stop_btn.setObjectName("dangerAction")
    self.test_qwen_stop_btn.setToolTip("停止当前部署位置对应的推理服务（远程含隧道）")
    service_row.addWidget(self.test_qwen_stop_btn)
    self.test_qwen_start_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.test_qwen_stop_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.test_qwen_refresh_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.test_open_studio_btn = _eai_platform_bootstrap.QPushButton("打开测试工作室…")
    self.test_open_studio_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.test_open_studio_btn.setToolTip(
        "另开独立前端窗口（非本 Tab）。等价于：bash test_studio/run_test_studio.sh"
    )
    self.test_open_studio_btn.clicked.connect(self._on_open_test_studio_clicked)
    service_row.addWidget(self.test_open_studio_btn)
    test_outer.addLayout(service_row)

    self.test_infer_log_edit = _eai_platform_bootstrap.QTextEdit()
    self.test_infer_log_edit.setReadOnly(True)
    self.test_infer_log_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    self.test_infer_log_edit.setPlaceholderText("推理服务日志…")
    self.test_infer_log_edit.setStyleSheet(
        f"QTextEdit {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; background-color: {_eai_configuration_policies.UI_BG_LOG}; "
        f"border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
    )
    self.test_infer_log_edit_resize_bar = _eai_ui_widgets_layouts._add_text_edit_with_drag_bar(
        test_outer,
        self.test_infer_log_edit,
        default_h=140,
        min_h=80,
        max_h=900,
        label="拖动调整推理日志高度",
        clearable=True,
    )
    # 推理部署后端在 chat_panel 创建后由 QwenDeployController 统一接管
    self._qwen_deploy = None  # type: ignore

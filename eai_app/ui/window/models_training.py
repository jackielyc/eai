"""ui.window.models training — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_discovery = module_ref('eai_app.adapters.models.discovery')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_perception_segmentation = module_ref('eai_app.perception.segmentation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_cameras_panels = module_ref('eai_app.ui.cameras.panels')

class ModelsTrainingMixin:
    def _on_train_browse_policy_clicked(self) -> None:
        if self._train_launcher.is_running():
            return
        current = self._get_psi_policy_dir()
        initial = current if current and _eai_platform_bootstrap.os.path.isdir(current) else _eai_platform_bootstrap.os.path.expanduser("~")
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(
            self,
            "选择 psi-policy 仓库目录",
            initial,
        )
        if not selected:
            return
        if not _eai_perception_segmentation.is_valid_psi_policy_dir(selected):
            _eai_platform_bootstrap.QMessageBox.warning(
                self,
                "无效目录",
                f"所选目录不是有效的 psi-policy 仓库：\n{selected}\n\n"
                "请确认目录下存在 psi_policy/train.py",
            )
            return
        self._psi_policy_dir_override = _eai_platform_bootstrap.os.path.abspath(selected)
        self._refresh_train_config_combo()
        self._update_train_path_ui()
        self.status_bar.showMessage(f"已选择 psi-policy: {self._psi_policy_dir_override}")

    def _on_train_start_clicked(self) -> None:
        config_name = str(self.train_config_combo.currentData() or _eai_configuration_policies.PSI_POLICY_CONFIG_DEFAULT)
        self._train_launcher.start(
            config_name,
            self.train_datahouse_edit.text(),
            self.train_view_edit.text(),
            repo_dir=self._get_psi_policy_dir(),
            num_epochs=self.train_epochs_spin.value(),
            batch_size=self.train_batch_spin.value(),
            logging_mode=str(self.train_logging_combo.currentData() or "offline"),
            num_processes=self.train_gpu_spin.value(),
        )
        self._update_train_ui()

    def _on_train_stop_clicked(self) -> None:
        self._train_launcher.stop()

    def _on_train_clear_log_clicked(self) -> None:
        self.train_log_edit.clear()

    def _on_reward_status(self, msg: str) -> None:
        if hasattr(self, "reward_status_label"):
            self.reward_status_label.setText(msg)
        self.status_bar.showMessage(msg)

    def _on_reward_mode_changed(self, *_args) -> None:
        if not hasattr(self, "reward_stack"):
            return
        mode = str(self.reward_mode_combo.currentData() or "single")
        idx = {"single": 0, "pair": 1, "batch": 2, "teleop": 3}.get(mode, 0)
        self.reward_stack.setCurrentIndex(idx)
        if mode == "teleop":
            self.reward_run_btn.setText("启动 teleop")
            self.reward_image_size_edit.setText(
                self.reward_image_size_edit.text().strip() or "3,128,128"
            )
            if abs(self.reward_threshold_spin.value() - 0.5) < 1e-6:
                self.reward_threshold_spin.setValue(0.2)
        else:
            self.reward_run_btn.setText("运行评测")

    def _on_reward_root_browse(self) -> None:
        cur = self.reward_root_edit.text().strip() or _eai_configuration_tabs.RLINF_ROOT_DEFAULT
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择 RLinf 仓库", cur)
        if selected:
            self.reward_root_edit.setText(selected)
            self.reward_python_edit.setText(_eai_adapters_models_discovery.resolve_rlinf_python(selected))

    def _on_reward_ckpt_browse(self) -> None:
        cur = self.reward_ckpt_edit.text().strip()
        initial = (
            _eai_platform_bootstrap.os.path.dirname(cur)
            if cur and _eai_platform_bootstrap.os.path.exists(cur)
            else _eai_adapters_models_discovery.resolve_rlinf_root(self.reward_root_edit.text())
        )
        path, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择 reward 权重",
            initial,
            "Weights (*.pt *.pth *.safetensors *.bin);;All (*)",
        )
        if path:
            self.reward_ckpt_edit.setText(path)

    def _on_reward_dataset_browse(self) -> None:
        cur = self.reward_dataset_edit.text().strip()
        initial = _eai_platform_bootstrap.os.path.dirname(cur) if cur else _eai_adapters_models_discovery.resolve_rlinf_root(
            self.reward_root_edit.text()
        )
        path, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择 reward 数据集 .pt",
            initial,
            "PyTorch (*.pt *.pth);;All (*)",
        )
        if path:
            self.reward_dataset_edit.setText(path)

    def _on_reward_image_browse(self, edit: _eai_platform_bootstrap.QLineEdit) -> None:
        cur = edit.text().strip()
        initial = _eai_platform_bootstrap.os.path.dirname(cur) if cur else _eai_configuration_models.EAI_DIR
        path, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择图像",
            initial,
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All (*)",
        )
        if path:
            edit.setText(path)

    def _reward_grab_camera_image(self, out_name: str) -> _eai_platform_bootstrap.Optional[str]:
        topic = str(self.reward_topic_combo.currentData() or "").strip()
        picked = None
        if topic:
            panel = self.panels.get(topic)
            if isinstance(panel, _eai_ui_cameras_panels.CameraPanel) and panel._latest_image is not None:
                picked = (topic, panel._latest_image.copy())
            else:
                cached = self._frame_cache.get(topic)
                if cached is not None and _eai_platform_bootstrap.np.asarray(cached).ndim >= 2:
                    picked = (topic, _eai_platform_bootstrap.np.asarray(cached).copy())
        if picked is None:
            src = self._pick_sam3_color_source()
            if src is None:
                self._append_reward_log("[ERROR] 无可用彩色图像")
                self._on_reward_status("无可用相机图像")
                return None
            picked = (src[0], src[1])
        topic, image = picked
        _eai_platform_bootstrap.os.makedirs(_eai_configuration_tabs.RLINF_REWARD_CACHE_DIR, exist_ok=True)
        path = _eai_platform_bootstrap.os.path.join(_eai_configuration_tabs.RLINF_REWARD_CACHE_DIR, out_name)
        bgr = _eai_platform_bootstrap.np.asarray(image)
        if bgr.ndim == 2:
            bgr = _eai_platform_bootstrap.cv2.cvtColor(bgr, _eai_platform_bootstrap.cv2.COLOR_GRAY2BGR)
        _eai_platform_bootstrap.cv2.imwrite(path, bgr)
        self._append_reward_log(f"已从 {topic} 抓帧 -> {path}")
        return path

    def _on_reward_capture_slot(self, slot: str) -> None:
        path = self._reward_grab_camera_image(f"pair_{slot}.png")
        if not path:
            return
        if slot == "a":
            self.reward_pair_a_edit.setText(path)
            self.reward_pair_a_preview.set_source_pixmap(_eai_platform_bootstrap.QPixmap(path))
        else:
            self.reward_pair_b_edit.setText(path)
            self.reward_pair_b_preview.set_source_pixmap(_eai_platform_bootstrap.QPixmap(path))

    def _reward_common_kwargs(self) -> dict:
        return {
            "rlinf_root": self.reward_root_edit.text().strip(),
            "python_bin": self.reward_python_edit.text().strip(),
            "ckpt": self.reward_ckpt_edit.text().strip(),
            "model_type": str(self.reward_type_combo.currentData() or "resnet"),
            "arch": str(self.reward_arch_combo.currentData() or "resnet18"),
            "image_size": self.reward_image_size_edit.text().strip() or "3,224,224",
            "device": str(self.reward_device_combo.currentData() or "auto"),
            "threshold": float(self.reward_threshold_spin.value()),
            "task": self.reward_task_edit.text().strip(),
        }

    def _on_reward_run_clicked(self) -> None:
        if self._reward_launcher.is_running():
            return
        mode = str(self.reward_mode_combo.currentData() or "single")
        common = self._reward_common_kwargs()
        if mode != "teleop" and not common["ckpt"]:
            self._append_reward_log("[ERROR] 请填写 ckpt 权重路径")
            self._on_reward_status("缺少 ckpt")
            return
        if mode == "teleop":
            self._reward_launcher.start_teleop(
                rlinf_root=common["rlinf_root"],
                python_bin=common["python_bin"],
                config_name=self.reward_teleop_config_edit.text().strip()
                or "realworld_teleop",
                ckpt=common["ckpt"],
                threshold=common["threshold"],
                model_type=common["model_type"],
                arch=common["arch"],
                image_size=common["image_size"],
            )
            self._update_reward_ui()
            return

        if mode == "single":
            source = str(self.reward_single_source_combo.currentData() or "camera")
            if source == "camera":
                path = self._reward_grab_camera_image("single_input.png")
            else:
                path = _eai_platform_bootstrap.os.path.abspath(
                    _eai_platform_bootstrap.os.path.expanduser(self.reward_single_path_edit.text().strip())
                )
                if not _eai_platform_bootstrap.os.path.isfile(path):
                    self._append_reward_log(f"[ERROR] 图像不存在: {path}")
                    return
            self.reward_single_preview.set_source_pixmap(_eai_platform_bootstrap.QPixmap(path))
            self.reward_score_label.setText("score: …")
            self._reward_launcher.start_score(mode="single", image=path, **common)
        elif mode == "pair":
            path_a = self.reward_pair_a_edit.text().strip()
            path_b = self.reward_pair_b_edit.text().strip()
            if str(self.reward_pair_source_combo.currentData() or "") == "camera":
                if not path_a:
                    path_a = self._reward_grab_camera_image("pair_a.png") or ""
                    if path_a:
                        self.reward_pair_a_edit.setText(path_a)
                if not path_b:
                    self._append_reward_log(
                        "[ERROR] 请先点「抓 B」再运行（或填本地路径）"
                    )
                    return
            path_a = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(path_a))
            path_b = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(path_b))
            if not _eai_platform_bootstrap.os.path.isfile(path_a) or not _eai_platform_bootstrap.os.path.isfile(path_b):
                self._append_reward_log("[ERROR] 图 A/B 路径无效")
                return
            self.reward_pair_a_preview.set_source_pixmap(_eai_platform_bootstrap.QPixmap(path_a))
            self.reward_pair_b_preview.set_source_pixmap(_eai_platform_bootstrap.QPixmap(path_b))
            self.reward_pair_result_label.setText("A/B: …")
            self._reward_launcher.start_score(
                mode="pair", image_a=path_a, image_b=path_b, **common
            )
        else:
            dataset = self.reward_dataset_edit.text().strip()
            if not dataset or not _eai_platform_bootstrap.os.path.isfile(
                _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(dataset))
            ):
                self._append_reward_log("[ERROR] 请选择有效的 .pt 数据集")
                return
            self.reward_batch_summary.setText("批量评测运行中…")
            self._reward_launcher.start_score(
                mode="batch",
                dataset=dataset,
                max_samples=int(self.reward_max_samples_spin.value()),
                **common,
            )
        self._update_reward_ui()

    def _on_reward_stop_clicked(self) -> None:
        self._append_reward_log("--- 用户停止 Reward 评测 ---")
        self._reward_launcher.stop()
        self._update_reward_ui()

    def _on_reward_result(self, result: dict) -> None:
        if not result.get("ok"):
            return
        mode = str(result.get("mode") or "")
        if mode == "single":
            score = result.get("score")
            self.reward_score_label.setText(
                f"score: {float(score):.4f}" if score is not None else "score: —"
            )
            img = str(result.get("image") or "")
            if img and _eai_platform_bootstrap.os.path.isfile(img):
                self.reward_single_preview.set_source_pixmap(_eai_platform_bootstrap.QPixmap(img))
            self._append_reward_log(
                f"单帧 score={score} infer={result.get('infer_s')}s "
                f"device={result.get('device')}"
            )
        elif mode == "pair":
            sa, sb = result.get("score_a"), result.get("score_b")
            prefer = result.get("prefer")
            delta = result.get("delta")
            self.reward_pair_result_label.setText(
                f"A={float(sa):.4f}  B={float(sb):.4f}\n"
                f"prefer={prefer}  Δ={float(delta):+.4f}"
            )
            self._append_reward_log(
                f"A/B prefer={prefer} A={sa} B={sb} Δ={delta} "
                f"infer={result.get('infer_s')}s"
            )
        elif mode == "batch":
            lines = [
                f"samples={result.get('num_samples')}/{result.get('num_total')}",
                f"accuracy={result.get('accuracy'):.4f}  "
                f"correct={result.get('correct')}",
                f"threshold={result.get('threshold')}  "
                f"mean={result.get('mean_score')}",
                f"pos_mean={result.get('pos_mean')}  "
                f"neg_mean={result.get('neg_mean')}",
                f"TP={result.get('tp')} TN={result.get('tn')} "
                f"FP={result.get('fp')} FN={result.get('fn')}",
                f"infer={result.get('infer_s')}s device={result.get('device')}",
            ]
            self.reward_batch_summary.setText("\n".join(lines))
            self._append_reward_log(
                f"批量 accuracy={result.get('accuracy'):.4f} "
                f"n={result.get('num_samples')}"
            )
        elif mode == "teleop":
            self._append_reward_log("teleop 进程正常结束")

    def _on_reward_wf_status(self, msg: str) -> None:
        if hasattr(self, "reward_wf_status_label"):
            self.reward_wf_status_label.setText(msg or "空闲")
        if hasattr(self, "status_bar") and self.status_bar is not None:
            self.status_bar.showMessage(msg)

    def _on_reward_wf_log_dir(self, path: str) -> None:
        if hasattr(self, "reward_wf_log_dir_edit"):
            self.reward_wf_log_dir_edit.setText(path)
        if hasattr(self, "reward_wf_open_log_btn"):
            self.reward_wf_open_log_btn.setEnabled(bool(path) and _eai_platform_bootstrap.os.path.isdir(path))

    def _on_reward_wf_root_browse(self) -> None:
        cur = self.reward_wf_root_edit.text().strip() or _eai_configuration_tabs.RLINF_ROOT_DEFAULT
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择 RLinf 仓库", cur)
        if selected:
            self.reward_wf_root_edit.setText(selected)
            self.reward_wf_python_edit.setText(_eai_adapters_models_discovery.resolve_rlinf_python(selected))
            self._refresh_reward_wf_configs()

    def _on_reward_wf_raw_browse(self) -> None:
        cur = self.reward_wf_raw_edit.text().strip() or _eai_platform_bootstrap.os.getcwd()
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择原始采集数据目录", cur)
        if selected:
            self.reward_wf_raw_edit.setText(selected)

    def _on_reward_wf_out_browse(self) -> None:
        cur = self.reward_wf_out_edit.text().strip() or _eai_platform_bootstrap.os.getcwd()
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择预处理输出目录", cur)
        if selected:
            self.reward_wf_out_edit.setText(selected)

    def _on_reward_wf_train_pt_browse(self) -> None:
        cur = self.reward_wf_train_pt_edit.text().strip() or _eai_platform_bootstrap.os.getcwd()
        path, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择 train 数据集",
            cur,
            "Datasets (*.pt *.jsonl);;All (*)",
        )
        if path:
            self.reward_wf_train_pt_edit.setText(path)

    def _on_reward_wf_val_pt_browse(self) -> None:
        cur = self.reward_wf_val_pt_edit.text().strip() or _eai_platform_bootstrap.os.getcwd()
        path, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择 val 数据集",
            cur,
            "Datasets (*.pt *.jsonl);;All (*)",
        )
        if path:
            self.reward_wf_val_pt_edit.setText(path)

    def _on_reward_wf_mode_changed(self, *_args) -> None:
        if not hasattr(self, "reward_wf_stack"):
            return
        mode = str(self.reward_wf_mode_combo.currentData() or "preprocess")
        if mode in ("preprocess", "pipeline"):
            self.reward_wf_stack.setCurrentIndex(0)
        elif mode == "train":
            self.reward_wf_stack.setCurrentIndex(1)
        else:
            self.reward_wf_stack.setCurrentIndex(2)
        self._refresh_reward_wf_configs()

    def _on_reward_wf_start_clicked(self) -> None:
        if self._reward_wf_launcher.is_running():
            self._on_reward_wf_status("Reward 工作流正在运行")
            return
        mode = str(self.reward_wf_mode_combo.currentData() or "preprocess")
        ds = str(self.reward_wf_dataset_combo.currentData() or "resnet")
        cfg = (
            str(self.reward_wf_config_combo.currentData() or "").strip()
            or self.reward_wf_config_combo.currentText().strip()
        )
        overrides_raw = self.reward_wf_overrides_edit.text().strip()
        user_overrides = _eai_platform_bootstrap.shlex.split(overrides_raw) if overrides_raw else []
        overrides = list(user_overrides)
        if mode == "train":
            train_pt = self.reward_wf_train_pt_edit.text().strip()
            val_pt = self.reward_wf_val_pt_edit.text().strip()
            if train_pt:
                overrides.append(f"data.train_data_paths={train_pt}")
            if val_pt:
                overrides.append(f"data.val_data_paths={val_pt}")
        self.reward_wf_log_edit.clear()
        self.reward_wf_log_dir_edit.clear()
        self.reward_wf_open_log_btn.setEnabled(False)
        self._reward_wf_launcher.start(
            mode=mode,
            rlinf_root=self.reward_wf_root_edit.text().strip(),
            python_bin=self.reward_wf_python_edit.text().strip(),
            config_name=cfg,
            dataset_type=ds,
            raw_data_path=self.reward_wf_raw_edit.text().strip(),
            output_dir=self.reward_wf_out_edit.text().strip(),
            num_samples=int(self.reward_wf_samples_spin.value()),
            val_split=float(self.reward_wf_val_spin.value()),
            fail_success_ratio=float(self.reward_wf_ratio_spin.value()),
            seed=int(self.reward_wf_seed_spin.value()),
            keep_last_frame=bool(self.reward_wf_keep_last_check.isChecked()),
            cuda_devices=self.reward_wf_cuda_edit.text().strip(),
            hydra_overrides=overrides,
        )
        self._update_reward_wf_ui()

    def _on_reward_wf_stop_clicked(self) -> None:
        self._append_reward_wf_log("--- 用户停止 Reward 工作流 ---")
        self._reward_wf_launcher.stop()
        self._update_reward_wf_ui()

    def _on_reward_wf_open_log_clicked(self) -> None:
        path = self.reward_wf_log_dir_edit.text().strip()
        if path and _eai_platform_bootstrap.os.path.isdir(path):
            _eai_platform_bootstrap.QDesktopServices.openUrl(_eai_platform_bootstrap.QUrl.fromLocalFile(path))

    def _on_dojo_rl_status(self, msg: str) -> None:
        if hasattr(self, "dojo_rl_status_label"):
            self.dojo_rl_status_label.setText(msg or "空闲")
        if hasattr(self, "status_bar") and self.status_bar is not None:
            self.status_bar.showMessage(msg)

    def _on_dojo_rl_log_dir(self, path: str) -> None:
        if hasattr(self, "dojo_rl_log_dir_edit"):
            self.dojo_rl_log_dir_edit.setText(path)
        if hasattr(self, "dojo_rl_open_log_btn"):
            self.dojo_rl_open_log_btn.setEnabled(bool(path) and _eai_platform_bootstrap.os.path.isdir(path))

    def _on_dojo_rl_root_browse(self) -> None:
        cur = self.dojo_rl_root_edit.text().strip() or _eai_configuration_policies.RISE_ROOT_DEFAULT
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择 RISE 仓库根目录", cur)
        if selected:
            self.dojo_rl_root_edit.setText(selected)
            self.dojo_rl_python_edit.setText(_eai_adapters_models_discovery.resolve_rise_python(selected))
            self._refresh_dojo_rl_configs()

    def _on_dojo_rl_model_browse(self) -> None:
        cur = self.dojo_rl_model_dir_edit.text().strip() or _eai_platform_bootstrap.os.getcwd()
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择策略 ckpt 目录", cur)
        if selected:
            self.dojo_rl_model_dir_edit.setText(selected)

    def _on_dojo_rl_reward_browse(self) -> None:
        cur = self.dojo_rl_reward_ckpt_edit.text().strip() or _eai_platform_bootstrap.os.getcwd()
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择奖励模型 ckpt 目录", cur)
        if selected:
            self.dojo_rl_reward_ckpt_edit.setText(selected)

    def _on_dojo_rl_resume_browse(self) -> None:
        cur = self.dojo_rl_resume_edit.text().strip() or _eai_platform_bootstrap.os.getcwd()
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择 resume 目录", cur)
        if selected:
            self.dojo_rl_resume_edit.setText(selected)

    def _on_dojo_rl_start_clicked(self) -> None:
        if self._dojo_rl_launcher.is_running():
            self._on_dojo_rl_status("仿真强化学习训练 正在运行")
            return
        mode = str(self.dojo_rl_mode_combo.currentData() or "single")
        cfg = (
            str(self.dojo_rl_config_combo.currentData() or "").strip()
            or self.dojo_rl_config_combo.currentText().strip()
        )
        if not cfg:
            self._append_dojo_rl_log("[ERROR] 请选择 config")
            self._on_dojo_rl_status("请选择 config")
            return
        overrides_raw = self.dojo_rl_overrides_edit.text().strip()
        overrides = _eai_platform_bootstrap.shlex.split(overrides_raw) if overrides_raw else []
        model_dir = self.dojo_rl_model_dir_edit.text().strip()
        reward_ckpt = self.dojo_rl_reward_ckpt_edit.text().strip()
        resume_dir = self.dojo_rl_resume_edit.text().strip()
        if not model_dir:
            self._append_dojo_rl_log(
                "[ERROR] 请填写策略 ckpt（rollout.model_dir），"
                "config 默认 path/to/policy/ckpt/steps 无法加载"
            )
            self._on_dojo_rl_status("缺少策略 ckpt")
            return
        if not _eai_platform_bootstrap.os.path.isdir(_eai_platform_bootstrap.os.path.expanduser(model_dir)):
            self._append_dojo_rl_log(f"[ERROR] 策略 ckpt 目录不存在: {model_dir}")
            self._on_dojo_rl_status("策略 ckpt 无效")
            return
        if not reward_ckpt:
            self._append_dojo_rl_log(
                "[WARN] 未填奖励 ckpt；若 config 仍是 path/to/reward/... "
                "加载奖励模型时会失败"
            )
        elif not _eai_platform_bootstrap.os.path.isdir(_eai_platform_bootstrap.os.path.expanduser(reward_ckpt)):
            self._append_dojo_rl_log(f"[ERROR] 奖励 ckpt 目录不存在: {reward_ckpt}")
            self._on_dojo_rl_status("奖励 ckpt 无效")
            return
        elif not _eai_adapters_models_discovery.is_rise_openpi_ckpt_dir(reward_ckpt):
            self._append_dojo_rl_log(
                f"[ERROR] 奖励 ckpt 目录缺少 model.safetensors: {reward_ckpt}"
            )
            self._on_dojo_rl_status("奖励 ckpt 无效")
            return
        elif _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(reward_ckpt)) == _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.expanduser(model_dir)
        ):
            self._append_dojo_rl_log(
                "[info] 奖励 ckpt 与策略相同（本机无独立 value_release；"
                "value head 可能随机初始化）"
            )
        if not _eai_adapters_models_discovery.is_rise_openpi_ckpt_dir(model_dir):
            self._append_dojo_rl_log(
                f"[ERROR] 策略 ckpt 目录缺少 model.safetensors: {model_dir}"
            )
            self._on_dojo_rl_status("策略 ckpt 无效")
            return
        if model_dir:
            overrides.append(f"rollout.model_dir={model_dir}")
        if reward_ckpt:
            overrides.append(
                f"actor.model.openpi.reward_model_ckpt={reward_ckpt}"
            )
        if resume_dir:
            overrides.append(f"runner.resume_dir={resume_dir}")
        cuda_devices = self.dojo_rl_cuda_edit.text().strip()
        auto_single = bool(self.dojo_rl_auto_gpu_check.isChecked())
        self.dojo_rl_log_edit.clear()
        self.dojo_rl_log_dir_edit.clear()
        self.dojo_rl_open_log_btn.setEnabled(False)
        n_gpu = _eai_adapters_models_discovery.count_visible_cuda_devices(cuda_devices)
        if auto_single and n_gpu == 1:
            self._append_dojo_rl_log(
                "[info] 检测到单卡，脚本将自动追加 placement/batch 覆盖"
            )
        self._dojo_rl_launcher.start(
            mode=mode,
            config_name=cfg,
            rise_root=self.dojo_rl_root_edit.text().strip(),
            python_bin=self.dojo_rl_python_edit.text().strip(),
            cuda_devices=cuda_devices,
            hydra_overrides=overrides,
            auto_single_gpu=auto_single,
        )
        self._update_dojo_rl_ui()

    def _on_dojo_rl_stop_clicked(self) -> None:
        self._append_dojo_rl_log("--- 用户停止 仿真强化学习训练 ---")
        self._dojo_rl_launcher.stop()
        self._update_dojo_rl_ui()

    def _on_dojo_rl_open_log_clicked(self) -> None:
        path = self.dojo_rl_log_dir_edit.text().strip()
        if path and _eai_platform_bootstrap.os.path.isdir(path):
            _eai_platform_bootstrap.QDesktopServices.openUrl(_eai_platform_bootstrap.QUrl.fromLocalFile(path))

    def _on_robometer_status(self, msg: str) -> None:
        if hasattr(self, "robometer_status_label"):
            self.robometer_status_label.setText(msg or "空闲")
        if hasattr(self, "status_bar") and self.status_bar is not None:
            self.status_bar.showMessage(msg)

    def _on_robometer_log_dir(self, path: str) -> None:
        if hasattr(self, "robometer_log_dir_edit"):
            self.robometer_log_dir_edit.setText(path)
        if hasattr(self, "robometer_open_log_btn"):
            self.robometer_open_log_btn.setEnabled(
                bool(path) and _eai_platform_bootstrap.os.path.isdir(path)
            )

    def _on_robometer_root_browse(self) -> None:
        cur = self.robometer_root_edit.text().strip() or _eai_configuration_tabs.ROBOMETER_ROOT_DEFAULT
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(
            self, "选择 robometer-policy-learning 仓库根目录", cur
        )
        if selected:
            self.robometer_root_edit.setText(selected)
            self.robometer_python_edit.setText(_eai_adapters_models_discovery.resolve_robometer_python(selected))
            self._on_robometer_job_changed()

    def _on_robometer_ckpt_browse(self) -> None:
        cur = self.robometer_ckpt_edit.text().strip() or _eai_platform_bootstrap.os.getcwd()
        start = cur if _eai_platform_bootstrap.os.path.isdir(cur) else _eai_platform_bootstrap.os.path.dirname(cur) or _eai_platform_bootstrap.os.getcwd()
        selected, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self, "选择 checkpoint", start, "Checkpoint (*.pt *.pth *.ckpt *);;All (*)"
        )
        if selected:
            self.robometer_ckpt_edit.setText(selected)

    def _robometer_job_defaults(self) -> _eai_platform_bootstrap.Tuple[str, str]:
        data = self.robometer_job_combo.currentData()
        if isinstance(data, tuple) and len(data) == 2:
            return str(data[0]), str(data[1])
        return "train", "libero_online_rl"

    def _on_robometer_job_changed(self, *_args) -> None:
        _job, default_cfg = self._robometer_job_defaults()
        self._refresh_robometer_configs(prefer=default_cfg)
        if hasattr(self, "robometer_start_btn"):
            if _job.startswith("eval"):
                self.robometer_start_btn.setText("启动评测")
            elif _job == "relabel_server":
                self.robometer_start_btn.setText("启动服务")
            else:
                self.robometer_start_btn.setText("启动训练")

    def _on_robometer_start_clicked(self) -> None:
        if self._robometer_launcher.is_running():
            self._on_robometer_status("RoboMeter 正在运行")
            return
        job, _default_cfg = self._robometer_job_defaults()
        cfg = (
            str(self.robometer_config_combo.currentData() or "").strip()
            or self.robometer_config_combo.currentText().strip()
        )
        if not cfg:
            self._append_robometer_log("[ERROR] 请选择 config")
            self._on_robometer_status("请选择 config")
            return
        overrides_raw = self.robometer_overrides_edit.text().strip()
        overrides = _eai_platform_bootstrap.shlex.split(overrides_raw) if overrides_raw else []
        ckpt = self.robometer_ckpt_edit.text().strip()
        if ckpt:
            if job.startswith("eval"):
                overrides.append(f"policy_checkpoint={ckpt}")
            else:
                overrides.append(f"training.load_dir={ckpt}")
        if bool(self.robometer_wandb_offline_check.isChecked()):
            overrides.append("logging.wandb_offline=true")
        render_gui = bool(self.robometer_render_gui_check.isChecked())
        if render_gui and not (_eai_platform_bootstrap.os.environ.get("DISPLAY") or "").strip():
            self._append_robometer_log(
                "[robometer] 警告: DISPLAY 为空，图形窗口可能无法弹出"
            )
        self.robometer_log_edit.clear()
        self.robometer_log_dir_edit.clear()
        self.robometer_open_log_btn.setEnabled(False)
        self._robometer_launcher.start(
            job=job,
            config_name=cfg,
            repo_root=self.robometer_root_edit.text().strip(),
            python_bin=self.robometer_python_edit.text().strip(),
            cuda_devices=self.robometer_cuda_edit.text().strip(),
            render_gui=render_gui,
            hydra_overrides=overrides,
        )
        self._update_robometer_ui()

    def _on_robometer_stop_clicked(self) -> None:
        self._append_robometer_log("--- 用户停止 RoboMeter ---")
        self._robometer_launcher.stop()
        self._update_robometer_ui()

    def _on_robometer_open_log_clicked(self) -> None:
        path = self.robometer_log_dir_edit.text().strip()
        if path and _eai_platform_bootstrap.os.path.isdir(path):
            _eai_platform_bootstrap.QDesktopServices.openUrl(_eai_platform_bootstrap.QUrl.fromLocalFile(path))

    def _on_roboicl_status(self, msg: str) -> None:
        if hasattr(self, "roboicl_status_label"):
            self.roboicl_status_label.setText(msg or "空闲")
        if hasattr(self, "status_bar") and self.status_bar is not None:
            self.status_bar.showMessage(msg)

    def _on_roboicl_log_dir(self, path: str) -> None:
        if hasattr(self, "roboicl_log_dir_edit"):
            self.roboicl_log_dir_edit.setText(path)
        if hasattr(self, "roboicl_open_log_btn"):
            self.roboicl_open_log_btn.setEnabled(
                bool(path) and _eai_platform_bootstrap.os.path.isdir(path)
            )

    def _on_roboicl_root_browse(self) -> None:
        cur = self.roboicl_root_edit.text().strip() or _eai_configuration_tabs.ROBOICL_ROOT_DEFAULT
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(
            self, "选择 RoboICL 仓库根目录", cur
        )
        if selected:
            self.roboicl_root_edit.setText(selected)
            self.roboicl_policy_python_edit.setText(
                _eai_adapters_models_discovery.resolve_roboicl_policy_python(selected)
            )
            self.roboicl_sim_python_edit.setText(
                _eai_adapters_models_discovery.resolve_roboicl_sim_python(selected)
            )
            if not self.roboicl_data_edit.text().strip():
                self.roboicl_data_edit.setText(_eai_platform_bootstrap.os.path.join(selected, "data"))
            if not self.roboicl_results_edit.text().strip():
                self.roboicl_results_edit.setText(_eai_platform_bootstrap.os.path.join(selected, "results"))
            self._refresh_roboicl_tasks()
            self._on_roboicl_protocol_changed()

    def _on_roboicl_data_browse(self) -> None:
        cur = self.roboicl_data_edit.text().strip() or _eai_configuration_tabs.ROBOICL_ROOT_DEFAULT
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择 RoboICL data 根目录", cur)
        if selected:
            self.roboicl_data_edit.setText(selected)

    def _on_roboicl_results_browse(self) -> None:
        cur = self.roboicl_results_edit.text().strip() or _eai_configuration_tabs.ROBOICL_ROOT_DEFAULT
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(
            self, "选择 RoboICL results 目录", cur
        )
        if selected:
            self.roboicl_results_edit.setText(selected)

    def _on_roboicl_reference_browse(self) -> None:
        cur = self.roboicl_reference_edit.text().strip() or self.roboicl_data_edit.text().strip()
        start = cur if _eai_platform_bootstrap.os.path.isdir(cur) else _eai_configuration_tabs.ROBOICL_ROOT_DEFAULT
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(
            self, "选择 one-shot reference 目录", start
        )
        if selected:
            self.roboicl_reference_edit.setText(selected)

    def _on_roboicl_mode_changed(self, *_args) -> None:
        mode = str(self.roboicl_mode_combo.currentData() or "run").strip()
        if hasattr(self, "roboicl_start_btn"):
            if mode == "prepare_assets":
                self.roboicl_start_btn.setText("准备 Assets")
            elif mode == "dry_run":
                self.roboicl_start_btn.setText("启动预检")
            elif mode == "capture_only":
                self.roboicl_start_btn.setText("启动相机探针")
            else:
                self.roboicl_start_btn.setText("启动 rollout")

    def _on_roboicl_protocol_changed(self, *_args) -> None:
        data = self.roboicl_protocol_combo.currentData()
        if not isinstance(data, tuple) or len(data) != 2:
            return
        rel_path, shots = data
        if hasattr(self, "roboicl_shots_spin"):
            self.roboicl_shots_spin.setValue(int(shots))
        root = (
            self.roboicl_root_edit.text().strip()
            if hasattr(self, "roboicl_root_edit")
            else _eai_configuration_tabs.ROBOICL_ROOT_DEFAULT
        )
        # profile path is resolved at start; keep relative hint in status only
        _ = _eai_platform_bootstrap.os.path.join(root, str(rel_path))

    def _on_roboicl_start_clicked(self) -> None:
        if self._roboicl_launcher.is_running():
            self._on_roboicl_status("RoboICL 正在运行")
            return
        mode = str(self.roboicl_mode_combo.currentData() or "run").strip()
        task = (
            str(self.roboicl_task_combo.currentData() or "").strip()
            or self.roboicl_task_combo.currentText().strip()
        )
        if not task:
            self._append_roboicl_log("[ERROR] 请选择任务")
            self._on_roboicl_status("请选择任务")
            return
        proto = self.roboicl_protocol_combo.currentData()
        profile = ""
        root = self.roboicl_root_edit.text().strip() or _eai_configuration_tabs.ROBOICL_ROOT_DEFAULT
        if isinstance(proto, tuple) and len(proto) == 2:
            profile = _eai_platform_bootstrap.os.path.join(root, str(proto[0]))
        render_gui = bool(self.roboicl_render_gui_check.isChecked())
        if render_gui and mode != "dry_run" and not (_eai_platform_bootstrap.os.environ.get("DISPLAY") or "").strip():
            self._append_roboicl_log(
                "[roboicl] 警告: DISPLAY 为空，图形窗口可能无法弹出"
            )
        if mode == "run" and not (
            self.roboicl_api_key_edit.text().strip()
            or _eai_platform_bootstrap.os.environ.get("ASTRA_API_KEY", "").strip()
        ):
            self._append_roboicl_log(
                "[roboicl] 警告: 未设置 ASTRA_API_KEY，完整 rollout 可能失败"
            )
        extras_raw = self.roboicl_extra_edit.text().strip()
        extras = _eai_platform_bootstrap.shlex.split(extras_raw) if extras_raw else []
        self.roboicl_log_edit.clear()
        self.roboicl_log_dir_edit.clear()
        self.roboicl_open_log_btn.setEnabled(False)
        self._roboicl_launcher.start(
            mode=mode,
            task=task,
            shots=int(self.roboicl_shots_spin.value()),
            seed=int(self.roboicl_seed_spin.value()),
            layout=int(self.roboicl_layout_spin.value()),
            gpu=int(self.roboicl_gpu_spin.value()),
            repo_root=root,
            policy_python=self.roboicl_policy_python_edit.text().strip(),
            sim_python=self.roboicl_sim_python_edit.text().strip(),
            profile=profile,
            data_root=self.roboicl_data_edit.text().strip(),
            results_root=self.roboicl_results_edit.text().strip(),
            reference=self.roboicl_reference_edit.text().strip(),
            model=self.roboicl_model_edit.text().strip(),
            api_key=self.roboicl_api_key_edit.text().strip(),
            render_gui=render_gui,
            extra_args=extras,
        )
        self._update_roboicl_ui()

    def _on_roboicl_stop_clicked(self) -> None:
        self._append_roboicl_log("--- 用户停止 RoboICL ---")
        self._roboicl_launcher.stop()
        self._update_roboicl_ui()

    def _on_roboicl_open_log_clicked(self) -> None:
        path = self.roboicl_log_dir_edit.text().strip()
        if path and _eai_platform_bootstrap.os.path.isdir(path):
            _eai_platform_bootstrap.QDesktopServices.openUrl(_eai_platform_bootstrap.QUrl.fromLocalFile(path))


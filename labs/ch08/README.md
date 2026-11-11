# labs/ch08 — дообучение

Полные версии листингов и опыты главы 8 «Дообучение». Короткие файлы самостоятельны, считаются на CPU за
секунды и выполняются ещё и в ноутбуке главы `notebooks/08_lora_sft.ipynb`. Запуск из корня репозитория:
`python labs/ch08/<файл>.py`.

| Файл | Раздел книги |
|---|---|
| `ft_kit.py` | 8.1–8.3: шаблон, маска, LoRA и NF4 одним модулем; его импортируют ноутбук и `booklm/experiments/ch08.py` |
| `sft_chat_template.py`, `sft_loss_mask.py`, `sft_packing.py` | 8.1 «Дообучение с учителем»: шаблон диалога, маска ошибки, упаковка |
| `lora_linear.py`, `lora_param_count.py`, `lora_train_merge.py` | 8.2 «Низкоранговая адаптация (LoRA)» |
| `dora_booklm.py` | 8.2.8 «Адаптер с разложением весов (DoRA)» (около часа) |
| `qlora_nf4.py`, `qlora_double_quant.py`, `qlora_layer.py` | 8.3 «QLoRA» |
| `reward_model_bt.py`, `policy_gradient_baseline.py`, `ppo_kl_clip.py` | 8.4 «Предпочтения и RLHF» |
| `rlhf_optimum.py`, `dpo_loss_toy_lm.py`, `dpo_tabular.py` | 8.5 «Прямая оптимизация предпочтений (DPO)» |
| `merge_mini_barrier.py`, `merge_booklm.py` | 8.7 «Слияние моделей» (около 20 и около 35 минут) |
| `forgetting_linear.py`, `reward_hacking_curve.py`, `sycophancy_refusal.py` | 8.8 «Поломки дообучения и их измерение» |
| `prefs_checks.py` | 8.4–8.8: числа листингов о предпочтениях, DPO, дистилляции и поломках в виде `assert` |

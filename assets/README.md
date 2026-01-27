## assets/

Bu klasör README içinde gösterilecek **rapor görselleri** için ayrılmıştır.

Önerilen akış:

1. Eğitimleri çalıştır (TensorBoard event logları oluşsun)
2. PNG grafikleri üret:

```bash
python scripts/export_tb_report.py --logdir logs/tensorboard --outdir assets
```

3. İstersen sadece `assets/tb/**.png` dosyalarını repoya commit’le (logların kendisini commit’lemek zorunda değilsin).


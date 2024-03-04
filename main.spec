# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['CamShow.py'],
             pathex=['E:\\Codes\\PyQt\\camshow'],
             binaries=[],
             datas=[("D:\\miniconda3\\envs\\myproject\\Lib\\site-packages\\torch", "torch"),
                ("D:\\miniconda3\envs\\myproject\\Lib\\site-packages\\torchvision", "torchvision")],
             hiddenimports=['torch', 'torchvision'],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
for d in a.datas:
	if '_C.cp37-win_amd64.pyd' in d[0]:
		a.datas.remove(d)
		break
for d in a.datas:
	if '_C.pyd' in d[0]:
		a.datas.remove(d)
		break
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
          name='CamShow',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None , icon='logo\\logo.ico')
